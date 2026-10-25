# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run every arm at every forecast origin of every trial and record the results.

At each origin the arms run one after another on identical inputs, so a slow
drift of the host (clock, thermals, other load) affects them alike. Each arm's
scenarios at every configured horizon (``horizon_scenarios``) are generated in
one call timed with ``time.perf_counter``; for JointFM the time includes the
client's health request and the HTTP round trip, which is what a portfolio
user pays, and with the next day as the only horizon the call is one
``generate_cvar_data`` call. Every measurement also reads the arm's expected
returns, the ``CvarData.mean`` its rebalancing runs optimize on. The first call of each arm in a process pays for
imports, CUDA context and cuML start-up, or the first connection, and is
flagged as warm-up so summaries can leave it out. Each call writes one row
per horizon, all carrying the call's time.
"""

import argparse
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from tqdm import tqdm

from portfolio_optimization.cvar_utils import expected_returns
from portfolio_optimization.gpu import apply_gpu_selection, visible_gpus

from .arms import (
    AnyArm,
    Arm,
    IndependentArm,
    SwapArm,
    check_same_checkpoint,
    describe_arms,
    resolve_arms,
    scenario_settings,
    simulators,
    verify_jointfm_service,
)
from .config import ExperimentConfig, load_config
from .horizons import horizon_scenarios
from .independence import independent_scenarios
from .scores import PortfolioScores, ScenarioScores
from .swaps import swap_scenarios
from .table import (
    CALL_COLUMNS,
    KEY_COLUMNS,
    RunMetadata,
    merge_identity,
    read_results,
    write_results,
)
from .trials import RETURNS_SETTINGS, OriginInputs, Trial, draw_trial, origin_inputs

BASE_COLUMNS = (
    *KEY_COLUMNS,
    "fit_type",
    "device",
    "assets_hash",
    "seed",
    "num_scen",
    "seconds",
    "warmup",
)
BASE_MODULES = ("numpy", "pandas", "sklearn", "scipy", "pyarrow", "yaml", "tqdm")
# Seconds between progress-bar refreshes; a log file gets a line per refresh.
PROGRESS_INTERVAL_TERMINAL = 1.0
PROGRESS_INTERVAL_LOG = 60.0


class Measurement(Protocol):
    """A reading taken from one arm's forecast at one origin and horizon.

    The forecast is the arm's scenarios over the horizon, shape
    ``(scenarios, assets)``, and its expected returns, the next-day
    ``CvarData.mean`` of shape ``(assets,)`` that its rebalancing runs
    optimize on (``OriginScenarios.expected_returns``).

    ``columns`` names the values it returns; the runner refuses a reading
    whose keys differ, so every row of a results file has the same columns.
    ``parameters`` holds the JSON-ready constants the reading depends on; both
    are part of the run identity.
    """

    columns: tuple[str, ...]
    parameters: Mapping[str, object]

    def __call__(
        self,
        inputs: OriginInputs,
        horizon: int,
        samples: np.ndarray,
        expected_returns: np.ndarray,
    ) -> Mapping[str, float]: ...


def load_prices(path: Path) -> pd.DataFrame:
    """Read a price CSV with dates in the first column, oldest row first.

    Raises
    ------
    ValueError
        If the dates are not strictly increasing.
    """
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError(f"{path} dates must be strictly increasing.")
    return prices


def _cpu_model() -> str:
    system = platform.system()
    if system == "Linux":
        output = subprocess.run(
            ["lscpu"], check=True, capture_output=True, text=True
        ).stdout
        for line in output.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
        raise RuntimeError("lscpu reports no CPU model name.")
    if system == "Darwin":
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    raise RuntimeError(f"Unsupported platform {system!r}.")


def _gpu() -> dict[str, str]:
    """The first visible GPU, the one this process's cuML work runs on."""
    first = visible_gpus()[0]
    name, driver, uuid = (
        subprocess.run(
            [
                "nvidia-smi",
                f"--id={first}",
                "--query-gpu=name,driver_version,uuid",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split(", ")
    )
    return {"name": name, "driver": driver, "uuid": uuid}


def host_metadata(arms: Sequence[AnyArm]) -> dict:
    """Host the timings were measured on; the GPU only when a GPU arm runs here.

    A JointFM GPU arm computes in its service, which reports its own device,
    so only an in-process GPU simulator (cuML KDE) requires a local GPU.
    """
    in_process_gpu = any(
        simulator.device == "GPU" and simulator.fit_type != "jointfm"
        for arm in arms
        for simulator in simulators(arm)
    )
    return {
        "hostname": platform.node(),
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": _cpu_model(),
        "logical_cores": os.cpu_count(),
        "gpu": _gpu() if in_process_gpu else None,
    }


def package_versions(
    arms: Sequence[AnyArm], extra_modules: Sequence[str] = ()
) -> dict[str, str]:
    """Versions of the distributions behind the selected arms and ``extra_modules``.

    The ``jointfm-client`` version is recorded with the JointFM services by
    ``check_jointfm_service``.

    Raises
    ------
    ValueError
        If a module a selected arm needs is not installed.
    """
    modules = [*BASE_MODULES, *extra_modules]
    if any(
        simulator.fit_type == "kde" and simulator.device == "GPU"
        for arm in arms
        for simulator in simulators(arm)
    ):
        modules.append("cuml")
    distributions = importlib.metadata.packages_distributions()
    missing = [module for module in modules if module not in distributions]
    if missing:
        raise ValueError(f"The selected arms need {missing}, which are not installed.")
    return {
        distribution: importlib.metadata.version(distribution)
        for module in modules
        for distribution in distributions[module]
    }


def run_identity(
    config: ExperimentConfig,
    arms: Sequence[AnyArm],
    prices: pd.DataFrame,
    measurements: Sequence[Measurement],
    services: dict[str, dict],
) -> dict:
    """Everything a results file's rows depend on, apart from the trial count."""
    return {
        "prices": {
            "file": config.prices.name,
            "sha256": hashlib.sha256(config.prices.read_bytes()).hexdigest(),
            "shape": list(prices.shape),
        },
        "experiment": {
            "seed": config.seed,
            "origins": config.origins,
            "assets": config.assets,
            "look_back": config.look_back,
            "origin_history": config.origin_history,
            "forward": config.forward,
            "num_scen": config.num_scen,
            "horizons": list(config.horizons),
            "returns": RETURNS_SETTINGS.model_dump(mode="json"),
        },
        "arms": describe_arms(arms, num_scen=config.num_scen),
        "measurements": [
            {"columns": list(m.columns), "parameters": dict(m.parameters)}
            for m in measurements
        ],
        "host": host_metadata(arms),
        "packages": package_versions(arms),
        "services": services,
    }


def _check_measurement_columns(measurements: Sequence[Measurement]) -> None:
    columns = [column for m in measurements for column in m.columns]
    clashing = sorted({c for c in columns if columns.count(c) > 1 or c in BASE_COLUMNS})
    if clashing:
        raise ValueError(f"Measurement columns {clashing} are not unique.")


class OriginScenarios:
    """Every simulator's scenarios at one origin, each generated at most once.

    A derived arm reads its donors' scenarios here, so it adds no simulator
    call when its donors run in the same session; a donor whose rows an earlier
    session wrote is generated again, without a row, from the origin's seed.
    A JointFM simulator first checks, outside the timed call, that ``.env``
    still selects the service recorded for it in ``services``.
    """

    def __init__(
        self,
        inputs: OriginInputs,
        config: ExperimentConfig,
        services: Mapping[str, dict],
    ):
        self.inputs = inputs
        self.config = config
        self.services = services
        self.scenarios: dict[str, Mapping[int, np.ndarray]] = {}

    def generate(self, arm: Arm) -> tuple[Mapping[int, np.ndarray], float]:
        """``arm``'s scenarios at every horizon and the seconds the call took.

        Raises
        ------
        ValueError
            If the JointFM service changed since the session started.
        """
        if arm.fit_type == "jointfm":
            verify_jointfm_service(arm, self.services[arm.name])
        start = time.perf_counter()
        scenarios = horizon_scenarios(
            arm,
            self.inputs.returns_dict,
            num_scen=self.config.num_scen,
            seed=self.inputs.seed,
            horizons=self.config.horizons,
        )
        seconds = time.perf_counter() - start
        self.scenarios[arm.name] = scenarios
        return scenarios, seconds

    def of(self, arm: Arm) -> Mapping[int, np.ndarray]:
        """``arm``'s scenarios, generated now unless an earlier call did."""
        if arm.name not in self.scenarios:
            self.generate(arm)
        return self.scenarios[arm.name]

    def expected_returns(self, arm: AnyArm) -> np.ndarray:
        """``arm``'s ``CvarData.mean`` at this origin, as its rebalancing runs build it.

        A simulator follows the library's rule (``expected_returns``) on its
        next-day scenarios. A swap cell keeps its marginal donor's and the
        independence control its window's, as their scenario generators do.
        JointFM's scenario mean is read off the next day of its multi-horizon
        request, whose draws differ from those of the next-day-only request a
        rebalancing decision sends.
        """
        if isinstance(arm, SwapArm):
            return self.expected_returns(arm.marginals)
        if isinstance(arm, IndependentArm):
            return self.expected_returns(arm.window)
        settings = scenario_settings(
            arm, num_scen=self.config.num_scen, seed=self.inputs.seed
        )
        return expected_returns(self.inputs.returns_dict, settings, self.of(arm)[1])


def run_arm(
    arm: AnyArm,
    trial: Trial,
    inputs: OriginInputs,
    config: ExperimentConfig,
    measurements: Sequence[Measurement],
    origin_scenarios: OriginScenarios,
    *,
    warmup: bool,
) -> list[dict]:
    """Generate ``arm``'s scenarios at one origin, time the call, and measure them.

    A simulator's time is its one call; a swap cell's is its shuffle alone,
    and the independence control's its permutation alone, after their donors'
    scenarios are read from ``origin_scenarios``.

    Returns
    -------
    list of dict
        One row per horizon of ``config.horizons``, in that order; every row
        carries the time of the one call that generated all horizons.

    Raises
    ------
    ValueError
        If a measurement returns other keys than its ``columns``, or the
        JointFM service changed since the session started.
    """
    if isinstance(arm, SwapArm):
        marginals = origin_scenarios.of(arm.marginals)
        dependence = origin_scenarios.of(arm.dependence)
        start = time.perf_counter()
        scenarios = swap_scenarios(
            marginals, dependence, num_scen=config.num_scen, seed=inputs.seed
        )
        seconds = time.perf_counter() - start
    elif isinstance(arm, IndependentArm):
        window = origin_scenarios.of(arm.window)
        start = time.perf_counter()
        scenarios = independent_scenarios(window, seed=inputs.seed)
        seconds = time.perf_counter() - start
    else:
        scenarios, seconds = origin_scenarios.generate(arm)
    mean = origin_scenarios.expected_returns(arm)
    rows = []
    for horizon in config.horizons:
        samples = scenarios[horizon]
        row = {
            "trial": trial.index,
            "origin": inputs.origin,
            "arm": arm.name,
            "horizon": horizon,
            "fit_type": arm.fit_type,
            "device": arm.device,
            "assets_hash": trial.assets_hash,
            "seed": inputs.seed,
            "num_scen": samples.shape[0],
            "seconds": seconds,
            "warmup": warmup,
        }
        for measurement in measurements:
            values = dict(measurement(inputs, horizon, samples, mean))
            if set(values) != set(measurement.columns):
                raise ValueError(
                    f"A measurement declared {measurement.columns} but returned "
                    f"{tuple(values)}."
                )
            row.update(values)
        rows.append(row)
    return rows


def pending_work(
    prices: pd.DataFrame, config: ExperimentConfig, arms: Sequence[AnyArm], done: set
) -> list[tuple[Trial, int, list[AnyArm]]]:
    """Every (trial, origin) with arms not yet in the results, in run order.

    Trials are cheap seeded draws, so all are drawn up front and the progress
    bar knows how much of the run is already done, also when a run resumes.
    """
    work = []
    for index in range(config.trials):
        trial = draw_trial(
            prices,
            index,
            seed=config.seed,
            n_assets=config.assets,
            n_origins=config.origins,
            origin_history=config.origin_history,
            forward=config.forward,
        )
        for origin in trial.origins:
            pending = [
                arm
                for arm in arms
                if (trial.index, prices.index[origin], arm.name) not in done
            ]
            if pending:
                work.append((trial, origin, pending))
    return work


def run_experiment(
    config: ExperimentConfig,
    measurements: Sequence[Measurement] = (),
    command: Sequence[str] = (),
) -> pd.DataFrame:
    """Run every pending (trial, origin, arm) and append it to the results file.

    Rows already in ``config.output`` are skipped, so an interrupted run
    resumes where it stopped and a larger ``trials`` extends a finished
    one. A session may also add arms: running the configuration again with
    ``.env`` pointing at the JointFM service on the other device adds that
    device's JointFM arm. The file is rewritten after each origin.

    Parameters
    ----------
    config : ExperimentConfig
        The run.
    measurements : sequence of Measurement
        Readings recorded per row besides the timing.
    command : sequence of str
        Command line of this session, recorded in the file's metadata.

    Returns
    -------
    pd.DataFrame
        Every row of the results file after this session.

    Raises
    ------
    ValueError
        If the existing file was measured under another identity (data,
        parameters, measurements, host, or packages, or other settings or
        service for an arm it already holds), if the JointFM arms' services
        serve different checkpoints, if ``.env`` selects another JointFM
        service during the session, or if a window has missing or
        non-positive prices.
    """
    _check_measurement_columns(measurements)
    prices = load_prices(config.prices)
    arms, services = resolve_arms(config.arms)
    identity = run_identity(config, arms, prices, measurements, services)
    frame = None
    commands = [list(command)]
    if config.output.exists():
        frame, stored = read_results(config.output)
        identity = merge_identity(config.output, stored.identity, identity)
        commands = stored.commands + commands
    check_same_checkpoint(identity["services"])
    metadata = RunMetadata(identity=identity, commands=commands)
    done = (
        set()
        if frame is None
        else set(frame[list(CALL_COLUMNS)].itertuples(index=False, name=None))
    )
    work = pending_work(prices, config, arms, done)
    warmed_up = set()
    added = 0
    total = config.trials * config.origins
    progress = tqdm(
        work,
        desc="simulator comparison",
        unit="origin",
        total=total,
        initial=total - len(work),
        mininterval=PROGRESS_INTERVAL_TERMINAL
        if sys.stderr.isatty()
        else PROGRESS_INTERVAL_LOG,
    )
    for trial, origin, pending in progress:
        inputs = origin_inputs(
            prices,
            trial,
            origin,
            seed=config.seed,
            look_back=config.look_back,
            forward=config.forward,
        )
        rows = []
        seconds = {}
        origin_scenarios = OriginScenarios(inputs, config, services)
        for arm in pending:
            arm_rows = run_arm(
                arm,
                trial,
                inputs,
                config,
                measurements,
                origin_scenarios,
                warmup=arm.name not in warmed_up,
            )
            rows += arm_rows
            seconds[arm.name] = arm_rows[0]["seconds"]
            warmed_up.add(arm.name)
        new = pd.DataFrame(rows)
        frame = new if frame is None else pd.concat([frame, new], ignore_index=True)
        write_results(config.output, frame, metadata)
        added += len(rows)
        progress.set_postfix_str(
            f"trial {trial.index} "
            + " ".join(f"{arm}={value:.3f}s" for arm, value in seconds.items()),
            refresh=False,
        )
    print(
        f"Simulator comparison finished: {added} new rows, "
        f"{0 if frame is None else len(frame)} in {config.output}.",
        flush=True,
    )
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.simulator_comparison",
        description=(
            "Run the scenario simulators on identical random asset universes and "
            "forecast origins, timing each scenario-generation call."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration file (see experiments/simulator_comparison/configs/).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    # Before anything imports cuML, so the KDE GPU arm sees the GPU .env selects.
    apply_gpu_selection()
    run_experiment(
        config,
        measurements=(
            ScenarioScores(),
            PortfolioScores(
                config.portfolio_sizes, config.rebalancing.cvar, config.assets
            ),
        ),
        command=sys.argv if argv is None else ["main", *argv],
    )
    return 0

# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-against-GPU experiment: each rebalancing decision on each device, stage by stage.

A *decision* is one optimization of the rebalancing backtest that the
configuration's ``experiment`` sets up (``rebalancing.py``): JointFM draws
scenarios from the look-back window of the decision's date, and the Mean-CVaR
optimizer chooses the portfolio from them. One *session* per device runs every
decision of trials ``0 .. trials - 1`` through the library's
``rebalance_portfolio``, as a user's pipeline would:

- the GPU session with JointFM on a GPU service and ``gpu_solver`` on the
  GPU that service computes on;
- the CPU session with JointFM on a CPU service and ``cpu_solver``, in a
  process that sees no GPU.

``StageClock`` stands in for the backtest's scenario generator and decision
observer, which times each decision's two stages without changing the
library. The *generation* stage is the scenario generator's call, including
the HTTP round trip to the service. The *optimizer* stage runs from that
call's return to the observer's call, so it spans the CVXPY problem's setup
and its solve. A decision's total is the sum of the two.

After a trial's backtest, each session times its own device's
``generators`` on every decision's look-back window. The GPU session then
runs every ``paired_solvers`` solver on every decision's identical problem,
rebuilt from the scenarios, expected returns, and previous portfolio the
backtest optimized on, and the solver sweep: JointFM scenarios at every
count of ``sweep_num_scen`` for the trial's first ``sweep_decisions``
decisions, each solved by every paired solver.

Before its first trial, a session checks its environment
(``select_session_gpus``, ``session_record``). The JointFM service must be a
local process that listens on the port of ``JOINTFM_LOCAL_BASE_URL``. A GPU
session's service must compute on the GPU this process runs cuOpt on, and a
CPU session's service must set ``OMP_NUM_THREADS``, so its thread count is
explicit. The service's command line, thread settings, and CPU affinity are
recorded with the session. Every trial also records how many CPU cores the
host's other processes used while it ran (``foreign_cores``).

Results go to one Parquet file (``table.py``). A row's ``kind`` says what it
holds:

- ``decision``: one decision of a session's pipeline. ``generation_seconds``
  and ``optimizer_seconds`` hold its stages, and ``seconds`` their sum.
- ``generation``: one scenario-generation call. ``seconds`` holds the call,
  and ``num_scen`` the scenarios it returned. The pipeline's JointFM calls
  appear here as well.
- ``solve``: one paired solve. ``setup_seconds`` holds the problem's setup,
  ``seconds`` the solve, ``solver_seconds`` the time the solver reports, and
  ``objective``, ``weights``, and ``cash`` its solution.

Rows of the sweep have ``sweep`` set. The first row of each kind and name in
a process is flagged ``warmup``, because it pays for imports, CUDA start-up,
or the first connection. A session's trial is written once it is complete,
so a rerun resumes after the last complete trial of each session.
Run it as ``python -m experiments.simulator_comparison.devices``.
"""

import argparse
import dataclasses
import hashlib
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

from portfolio_optimization import cvar_utils, risk_free
from portfolio_optimization.gpu import (
    CUDA_VISIBLE_DEVICES,
    apply_gpu_selection,
    visible_gpus,
)
from portfolio_optimization.portfolio import Portfolio

from .arms import (
    ARMS_BY_NAME,
    Arm,
    Device,
    check_jointfm_service,
    scenario_settings,
    verify_jointfm_service,
)
from .config import DevicesConfig, ExperimentConfig, load_devices_config
from .rebalancing import (
    backtest_api_settings,
    calibrated_cvar_params,
    draw_rebalancing_trial,
    run_rebalancer,
    write_universe,
)
from .runner import (
    PROGRESS_INTERVAL_LOG,
    PROGRESS_INTERVAL_TERMINAL,
    gpu_info,
    host_metadata,
    load_prices,
    package_versions,
)
from .solvers import SOLVERS_BY_NAME, check_cuopt_methods, solver_settings
from .table import RunMetadata, merge_identity, read_results, write_results
from .trials import RETURNS_SETTINGS, rebalancing_seed

DEVICES: tuple[Device, ...] = ("GPU", "CPU")
KINDS = ("decision", "generation", "solve")
COLUMNS = (
    "kind",
    "session",
    "trial",
    "decision",
    "date",
    "name",
    "device",
    "solver",
    "sweep",
    "num_scen",
    "seconds",
    "generation_seconds",
    "optimizer_seconds",
    "setup_seconds",
    "solver_seconds",
    "objective",
    "cash",
    "weights",
    "warmup",
    "foreign_cores",
)
# The client selector a session's JointFM service is reached through; the
# Taskfile sets it per session over .env.
LOCAL_URL_VARIABLE = "JOINTFM_LOCAL_BASE_URL"
LOCAL_HOSTS = ("127.0.0.1", "localhost", "0.0.0.0", "::1")
# Environment of the service process that decides where and how it computes.
SERVICE_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    CUDA_VISIBLE_DEVICES,
    "JOINTFM_INFERENCE_DEVICE",
    "JOINTFM_INFERENCE_PRECISION",
    "JOINTFM_INFERENCE_MAX_SAMPLE_COUNT",
    "JOINTFM_INFERENCE_MAX_CONCURRENT_REQUESTS",
)
CPU_THREADS_VARIABLE = "OMP_NUM_THREADS"
# Distributions behind the solvers and the process records, beside the arms'.
EXTRA_MODULES = ("cvxpy", "psutil")


@dataclass(frozen=True)
class Decision:
    """One optimization of a backtest, as ``StageClock`` captured it.

    Attributes
    ----------
    date : pd.Timestamp
        The decision's date.
    window : dict
        The look-back window's returns dict before any scenarios were added,
        which a timed generator starts from.
    cash_return : float
        The cash return the decision's scenarios carry.
    returns : dict
        The window's returns dict with the ``CvarData`` the decision
        optimized on.
    chosen : Portfolio
        The portfolio the decision chose.
    previous : Portfolio or None
        The portfolio its turnover was measured from: the previous decision's
        choice, or ``None`` for the first, which the optimizer then measures
        from equal weights.
    generation_seconds, optimizer_seconds : float
        The decision's stages (module docstring).
    """

    date: pd.Timestamp
    window: dict
    cash_return: float
    returns: dict
    chosen: Portfolio
    previous: Portfolio | None
    generation_seconds: float
    optimizer_seconds: float


class StageClock:
    """Scenario generator and decision observer of one backtest that time its decisions.

    ``rebalance_portfolio`` calls ``generate`` for a decision's scenarios,
    sets up and solves the problem, and then calls ``observe`` with the chosen
    portfolio, so the time from ``generate``'s return to ``observe`` is the
    optimizer stage. Every optimization of a backtest, the buy-and-hold one
    on the window's first day included, measures turnover from the portfolio
    the previous one chose, which ``observe`` records as ``previous``.
    """

    def __init__(self, generator: Callable[..., dict] = cvar_utils.generate_cvar_data):
        self._generator = generator
        self._pending: tuple | None = None
        self.decisions: list[Decision] = []

    def generate(self, returns_dict: dict, settings, *, cash_return: float) -> dict:
        """Generate a decision's scenarios as ``generator`` does, timing the call.

        Raises
        ------
        RuntimeError
            If the previous call's decision was never observed.
        """
        if self._pending is not None:
            raise RuntimeError("Scenarios were generated twice without a decision.")
        # The generator adds cvar_data to the dict it receives.
        window = dict(returns_dict)
        start = time.perf_counter()
        returns = self._generator(returns_dict, settings, cash_return=cash_return)
        generated = time.perf_counter()
        self._pending = (window, cash_return, returns, generated - start, generated)
        return returns

    def observe(self, date: pd.Timestamp, cvar_data, chosen: Portfolio) -> None:
        """Record a decision, its optimizer stage ending now.

        Raises
        ------
        RuntimeError
            If the decision did not optimize on the last generated scenarios.
        """
        observed = time.perf_counter()
        if self._pending is None or cvar_data is not self._pending[2]["cvar_data"]:
            raise RuntimeError(
                "A decision optimized on scenarios its generator did not time."
            )
        window, cash_return, returns, generation, generated = self._pending
        self._pending = None
        self.decisions.append(
            Decision(
                date=date,
                window=window,
                cash_return=cash_return,
                returns=returns,
                chosen=chosen,
                previous=self.decisions[-1].chosen if self.decisions else None,
                generation_seconds=generation,
                optimizer_seconds=observed - generated,
            )
        )


def run_decisions(
    arm: Arm,
    index: int,
    params,
    experiment: ExperimentConfig,
    devices: DevicesConfig,
    universe_path: Path,
    window,
    risk_free_rates: pd.Series,
    solver: str,
) -> list[Decision]:
    """Every decision of trial ``index``'s backtest, with JointFM ``arm`` and ``solver``.

    Raises
    ------
    ValueError
        If the decisions recorded are not the window's first day and the
        re-optimization dates.
    """
    dates = load_prices(universe_path).index
    clock = StageClock()
    _, _, re_optimize_dates, _ = run_rebalancer(
        dates,
        universe_path,
        window,
        params,
        experiment.rebalancing,
        risk_free_rates,
        settings=scenario_settings(
            arm,
            num_scen=devices.num_scen,
            seed=rebalancing_seed(experiment.seed, index),
        ),
        generator=clock.generate,
        observer=clock.observe,
        solver=solver,
    )
    decided = [decision.date for decision in clock.decisions]
    if decided != [dates[window.start], *re_optimize_dates]:
        raise ValueError(
            f"The run recorded decisions on {decided}, but optimized on the "
            f"window's first day and on {re_optimize_dates}."
        )
    return clock.decisions


def time_generation(
    arm: Arm, decision: Decision, num_scen: int, seed: int
) -> tuple[dict, float]:
    """``arm``'s scenarios from a decision's look-back window, and the call's seconds."""
    settings = scenario_settings(arm, num_scen=num_scen, seed=seed)
    start = time.perf_counter()
    returns = cvar_utils.generate_cvar_data(
        dict(decision.window), settings, cash_return=decision.cash_return
    )
    return returns, time.perf_counter() - start


def solve_decision(decision: Decision, params, solver: str) -> dict:
    """Rebuild a decision's problem from its scenarios and previous portfolio, and solve it.

    Returns
    -------
    dict
        ``setup_seconds`` (the problem's construction), ``seconds`` (its
        solve), ``solver_seconds`` (as the solver reports it), and the
        solution's ``objective``, asset ``weights``, and ``cash``.
    """
    from portfolio_optimization import cvar_optimizer

    start = time.perf_counter()
    problem = cvar_optimizer.CVaR(
        returns_dict=decision.returns,
        cvar_params=params,
        api_settings=backtest_api_settings(),
        existing_portfolio=decision.previous,
    )
    built = time.perf_counter()
    result, chosen = problem.solve_optimization_problem(
        solver_settings(solver), print_results=False
    )
    solved = time.perf_counter()
    return {
        "setup_seconds": built - start,
        "seconds": solved - built,
        "solver_seconds": float(result["solve time"]),
        "objective": float(result["obj"]),
        "weights": np.asarray(chosen.weights, dtype=float).tolist(),
        "cash": float(chosen.cash),
    }


def decision_rows(
    arm: Arm, solver: str, decisions: Sequence[Decision], num_scen: int
) -> list[dict]:
    """A decision row and a generation row per pipeline decision."""
    rows = []
    for index, decision in enumerate(decisions):
        common = {"decision": index, "date": decision.date, "sweep": False}
        rows.append(
            {
                **common,
                "kind": "decision",
                "name": arm.name,
                "device": arm.device,
                "solver": solver,
                "num_scen": num_scen,
                "seconds": decision.generation_seconds + decision.optimizer_seconds,
                "generation_seconds": decision.generation_seconds,
                "optimizer_seconds": decision.optimizer_seconds,
            }
        )
        rows.append(
            {
                **common,
                "kind": "generation",
                "name": arm.name,
                "device": arm.device,
                "num_scen": decision.returns["cvar_data"].R.shape[1],
                "seconds": decision.generation_seconds,
            }
        )
    return rows


def generator_rows(
    arms: Sequence[Arm], decisions: Sequence[Decision], num_scen: int, seed: int
) -> list[dict]:
    """A generation row per decision and simulator of ``arms``."""
    rows = []
    for index, decision in enumerate(decisions):
        for arm in arms:
            returns, seconds = time_generation(arm, decision, num_scen, seed)
            rows.append(
                {
                    "kind": "generation",
                    "decision": index,
                    "date": decision.date,
                    "name": arm.name,
                    "device": arm.device,
                    "sweep": False,
                    "num_scen": returns["cvar_data"].R.shape[1],
                    "seconds": seconds,
                }
            )
    return rows


def solve_rows(
    decisions: Sequence[Decision],
    params,
    solvers: Sequence[str],
    *,
    first: int = 0,
    sweep: bool,
) -> list[dict]:
    """A solve row per decision and solver; decision indices start at ``first``."""
    rows = []
    for index, decision in enumerate(decisions, start=first):
        for name in solvers:
            rows.append(
                {
                    "kind": "solve",
                    "decision": index,
                    "date": decision.date,
                    "name": name,
                    "device": SOLVERS_BY_NAME[name].device,
                    "solver": name,
                    "sweep": sweep,
                    "num_scen": decision.returns["cvar_data"].R.shape[1],
                    **solve_decision(decision, params, name),
                }
            )
    return rows


def sweep_rows(
    arm: Arm,
    decisions: Sequence[Decision],
    params,
    devices: DevicesConfig,
    seed: int,
) -> list[dict]:
    """JointFM's generation and every paired solve at each sweep count, for the first decisions."""
    rows = []
    for index, decision in enumerate(decisions[: devices.sweep_decisions]):
        for count in devices.sweep_num_scen:
            returns, seconds = time_generation(arm, decision, count, seed)
            rows.append(
                {
                    "kind": "generation",
                    "decision": index,
                    "date": decision.date,
                    "name": arm.name,
                    "device": arm.device,
                    "sweep": True,
                    "num_scen": returns["cvar_data"].R.shape[1],
                    "seconds": seconds,
                }
            )
            swept = dataclasses.replace(decision, returns=returns)
            rows += solve_rows(
                [swept], params, devices.paired_solvers, first=index, sweep=True
            )
    return rows


def cpu_seconds_by_process(excluded: Sequence[int]) -> dict[int, float]:
    """User and system CPU seconds of every process apart from ``excluded`` and their children.

    A process that exits while it is read is skipped, as is one whose times
    this user may not read.
    """
    skipped = set(excluded)
    for pid in excluded:
        skipped |= {child.pid for child in psutil.Process(pid).children(recursive=True)}
    seconds = {}
    for process in psutil.process_iter(["pid", "cpu_times"]):
        times = process.info["cpu_times"]
        if process.info["pid"] in skipped or times is None:
            continue
        seconds[process.info["pid"]] = times.user + times.system
    return seconds


def foreign_cores(
    before: dict[int, float], after: dict[int, float], seconds: float
) -> float:
    """Mean CPU cores the processes of ``after`` used over ``seconds`` since ``before``.

    A process that started in between counts with all its CPU time; one that
    exited in between is no longer seen, so the reading can fall short.
    """
    used = sum(after[pid] - before.get(pid, 0.0) for pid in after)
    return used / seconds


def service_process(url: str) -> tuple[int, dict]:
    """The local process that serves ``url``, and what decides how it computes.

    Returns
    -------
    tuple
        Its process ID, and its record: its command line, the variables of
        ``SERVICE_ENVIRONMENT`` in its environment (``None`` when unset), and,
        on Linux, the number of CPUs it may run on.

    Raises
    ------
    ValueError
        If ``url`` names no local host and port, or not exactly one process of
        this user listens on the port.
    """
    parts = urlsplit(url)
    if parts.hostname not in LOCAL_HOSTS or parts.port is None:
        raise ValueError(
            f"{url} is no local service with a port; the experiment names the "
            f"hardware of local services only, so start each JointFM service on "
            f"this host."
        )
    pids = {
        connection.pid
        for connection in psutil.net_connections(kind="tcp")
        if connection.status == psutil.CONN_LISTEN
        and connection.laddr.port == parts.port
    }
    if len(pids) != 1 or None in pids:
        raise ValueError(
            f"Expected one process of this user listening on port {parts.port} of "
            f"{url}, found {sorted(pid for pid in pids if pid is not None)} and "
            f"{sum(pid is None for pid in pids)} of other users."
        )
    pid = pids.pop()
    process = psutil.Process(pid)
    environment = process.environ()
    return pid, {
        "command": process.cmdline(),
        "environment": {name: environment.get(name) for name in SERVICE_ENVIRONMENT},
        "usable_cpus": usable_cpus(process),
    }


def usable_cpus(process: psutil.Process) -> int | None:
    """The CPUs ``process`` may run on, from its affinity; ``None`` on macOS, which has none."""
    return len(process.cpu_affinity()) if platform.system() == "Linux" else None


def service_gpu(device: str, visible: str | None) -> str:
    """The physical GPU a service on torch ``device`` computes on, as an index or UUID.

    ``visible`` is the service's ``CUDA_VISIBLE_DEVICES``, ``None`` when unset;
    ``cuda`` without an index is the first visible GPU.

    Raises
    ------
    ValueError
        If the device's index exceeds the GPUs the service sees.
    """
    index = int(device.split(":")[1]) if ":" in device else 0
    if visible is None:
        return str(index)
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if index >= len(entries):
        raise ValueError(
            f"The service computes on {device} but sees only {CUDA_VISIBLE_DEVICES}="
            f"{visible!r}."
        )
    return entries[index]


def cpu_topology() -> dict:
    """Sockets, physical cores, threads per core, and NUMA nodes of this host's CPUs.

    Raises
    ------
    RuntimeError
        On a platform other than Linux and macOS, or when lscpu omits a field.
    """
    system = platform.system()
    if system == "Linux":
        output = subprocess.run(
            ["lscpu"], check=True, capture_output=True, text=True
        ).stdout
        fields = dict(
            (key.strip(), value.strip())
            for key, _, value in (line.partition(":") for line in output.splitlines())
        )
        wanted = (
            "Socket(s)",
            "Core(s) per socket",
            "Thread(s) per core",
            "NUMA node(s)",
        )
        missing = [key for key in wanted if key not in fields]
        if missing:
            raise RuntimeError(f"lscpu reports no {missing}.")
        sockets, cores, threads, nodes = (int(fields[key]) for key in wanted)
        return {
            "sockets": sockets,
            "physical_cores": sockets * cores,
            "threads_per_core": threads,
            "numa_nodes": nodes,
        }
    if system == "Darwin":

        def sysctl(name: str) -> int:
            return int(
                subprocess.run(
                    ["sysctl", "-n", name], check=True, capture_output=True, text=True
                ).stdout
            )

        physical = sysctl("hw.physicalcpu")
        return {
            "sockets": sysctl("hw.packages"),
            "physical_cores": physical,
            "threads_per_core": sysctl("hw.logicalcpu") // physical,
            # macOS shares one memory between all cores.
            "numa_nodes": 1,
        }
    raise RuntimeError(f"Unsupported platform {system!r}.")


def select_session_gpus(device: Device) -> str:
    """Apply the session's GPU selection before anything imports cvxpy or cuOpt.

    The CPU session must hide every GPU itself, through an empty
    ``CUDA_VISIBLE_DEVICES`` in its environment, because ``.env`` would
    otherwise select one; the GPU session takes ``.env``'s selection.

    Returns
    -------
    str
        The ``CUDA_VISIBLE_DEVICES`` value now in effect.

    Raises
    ------
    ValueError
        If the CPU session's environment does not hide every GPU, or the GPU
        session sees none.
    """
    if device == "CPU" and os.environ.get(CUDA_VISIBLE_DEVICES) != "":
        raise ValueError(
            f"The CPU session must hide every GPU: start it with {CUDA_VISIBLE_DEVICES} "
            f"set empty, as task experiments:devices does."
        )
    selection = apply_gpu_selection()
    if device == "GPU":
        visible_gpus()
    return selection


def session_record(device: Device, service: dict) -> tuple[int, dict]:
    """Check a session's environment and record it.

    ``service`` is the health record of the JointFM service the client
    selects (``check_jointfm_service``).

    Returns
    -------
    tuple
        The service's process ID, and the session's record: the service's
        health and process (``service_process``), this process's
        ``CUDA_VISIBLE_DEVICES`` and usable CPUs, and for the GPU session the
        GPU that cuOpt and the service compute on.

    Raises
    ------
    ValueError
        If ``JOINTFM_LOCAL_BASE_URL`` is unset, the service is not a single
        local process, a GPU session's service computes on another GPU than
        this process, or a CPU session's service leaves ``OMP_NUM_THREADS``
        unset.
    """
    if LOCAL_URL_VARIABLE not in os.environ:
        raise ValueError(
            f"{LOCAL_URL_VARIABLE} is not set; task experiments:devices sets it to "
            f"each session's service."
        )
    pid, process = service_process(os.environ[LOCAL_URL_VARIABLE])
    record = {
        "jointfm": service,
        "service_process": process,
        "harness": {
            CUDA_VISIBLE_DEVICES: os.environ[CUDA_VISIBLE_DEVICES],
            "usable_cpus": usable_cpus(psutil.Process()),
        },
    }
    if device == "GPU":
        own = gpu_info(visible_gpus()[0])
        served = gpu_info(
            service_gpu(service["device"], process["environment"][CUDA_VISIBLE_DEVICES])
        )
        if served["uuid"] != own["uuid"]:
            raise ValueError(
                f"The JointFM service computes on {served} but this process runs "
                f"cuOpt on {own}; select the service's GPU in {CUDA_VISIBLE_DEVICES}."
            )
        record["gpu"] = own
    elif process["environment"][CPU_THREADS_VARIABLE] is None:
        raise ValueError(
            f"The CPU service leaves {CPU_THREADS_VARIABLE} unset; start it with an "
            f"explicit thread count, so the results record how many threads it used."
        )
    return pid, record


def devices_identity(
    devices: DevicesConfig,
    experiment: ExperimentConfig,
    prices: pd.DataFrame,
    generators: Sequence[Arm],
) -> dict:
    """Everything a results file depends on apart from its sessions and trial count.

    Both sessions record the same entries, so either may resume the file the
    other started; ``arms`` and ``services`` collect each session's own.
    """
    rebalancing = experiment.rebalancing
    host = host_metadata(())
    # Each session records the GPU it saw.
    del host["gpu"]
    solver_modules = sorted(
        {SOLVERS_BY_NAME[name].module for name in devices.paired_solvers}
    )
    return {
        "prices": {
            "file": experiment.prices.name,
            "sha256": hashlib.sha256(experiment.prices.read_bytes()).hexdigest(),
            "shape": list(prices.shape),
        },
        "risk_free_rates": {
            "file": rebalancing.risk_free_rates.name,
            "sha256": hashlib.sha256(
                rebalancing.risk_free_rates.read_bytes()
            ).hexdigest(),
        },
        "experiment": {
            "seed": experiment.seed,
            "assets": experiment.assets,
            "origins": experiment.origins,
            "origin_history": experiment.origin_history,
            "forward": experiment.forward,
            "returns": RETURNS_SETTINGS.model_dump(mode="json"),
        },
        "rebalancing": rebalancing.model_dump(
            mode="json",
            exclude={"output", "scenario_results", "arms", "trials", "solver"},
        ),
        "devices": devices.model_dump(
            mode="json", exclude={"experiment", "output", "trials"}
        ),
        "host": {**host, **cpu_topology()},
        "packages": package_versions(generators, (*EXTRA_MODULES, *solver_modules)),
        "arms": {},
        "services": {},
    }


def _flag_warmups(rows: list[dict], seen: set[tuple[str, str]]) -> None:
    for row in rows:
        key = (row["kind"], row["name"])
        row["warmup"] = key not in seen
        seen.add(key)


def run_device_session(
    devices: DevicesConfig,
    experiment: ExperimentConfig,
    device: Device,
    command: Sequence[str] = (),
) -> pd.DataFrame:
    """Run every pending trial of one device's session and append it to the results file.

    Call ``select_session_gpus`` first. Trials this session already holds in
    ``devices.output`` are skipped, so a rerun resumes and a larger
    ``trials`` extends the file; a file measured under other conditions is
    refused (``table.merge_identity``).

    Raises
    ------
    ValueError
        If the JointFM service the client selects computes on another device,
        the session's environment fails ``session_record``'s checks, the file
        was measured under other conditions, the service changes during the
        session, or a backtest records other decisions than it optimized.
    """
    prices = load_prices(experiment.prices)
    rates = risk_free.load_risk_free_rates(experiment.rebalancing.risk_free_rates)
    jointfm, service = check_jointfm_service()
    if jointfm.device != device:
        raise ValueError(
            f"The JointFM service the client selects computes on the "
            f"{jointfm.device}; the {device} session needs one on the {device}."
        )
    generators = tuple(ARMS_BY_NAME[name] for name in devices.generators)
    own_generators = tuple(arm for arm in generators if arm.device == device)
    solver = devices.gpu_solver if device == "GPU" else devices.cpu_solver
    paired = devices.paired_solvers if device == "GPU" else ()
    if device == "GPU":
        check_cuopt_methods(devices.paired_solvers)
    service_pid, record = session_record(device, service)

    identity = devices_identity(devices, experiment, prices, generators)
    identity["arms"][jointfm.name] = {
        "device": device,
        "solver": solver,
        "generators": [arm.name for arm in own_generators],
        "paired_solvers": list(paired),
    }
    identity["services"][jointfm.name] = record
    frame = None
    commands = [list(command)]
    if devices.output.exists():
        frame, stored = read_results(devices.output)
        identity = merge_identity(devices.output, stored.identity, identity)
        commands = stored.commands + commands
    metadata = RunMetadata(identity=identity, commands=commands)
    done = (
        set()
        if frame is None
        else set(
            frame.loc[frame["kind"] == "decision", ["session", "trial"]].itertuples(
                index=False, name=None
            )
        )
    )
    work = [index for index in range(devices.trials) if (device, index) not in done]
    progress = tqdm(
        work,
        desc=f"{device} session",
        unit="trial",
        total=devices.trials,
        initial=devices.trials - len(work),
        mininterval=PROGRESS_INTERVAL_TERMINAL
        if sys.stderr.isatty()
        else PROGRESS_INTERVAL_LOG,
    )
    seen: set[tuple[str, str]] = set()
    excluded = (os.getpid(), service_pid)
    for index in progress:
        trial, window = draw_rebalancing_trial(prices, index, experiment)
        universe_path = write_universe(prices, trial, devices.universe_dir)
        params = calibrated_cvar_params(
            prices[list(trial.assets)], window, experiment.rebalancing
        )
        seed = rebalancing_seed(experiment.seed, index)
        before = cpu_seconds_by_process(excluded)
        start = time.perf_counter()
        verify_jointfm_service(jointfm, service)
        decisions = run_decisions(
            jointfm,
            index,
            params,
            experiment,
            devices,
            universe_path,
            window,
            rates,
            solver,
        )
        rows = decision_rows(jointfm, solver, decisions, devices.num_scen)
        rows += generator_rows(own_generators, decisions, devices.num_scen, seed)
        if device == "GPU":
            rows += solve_rows(decisions, params, paired, sweep=False)
            rows += sweep_rows(jointfm, decisions, params, devices, seed)
        verify_jointfm_service(jointfm, service)
        foreign = foreign_cores(
            before, cpu_seconds_by_process(excluded), time.perf_counter() - start
        )
        _flag_warmups(rows, seen)
        new = pd.DataFrame(rows, columns=list(COLUMNS)).assign(
            session=device, trial=index, foreign_cores=foreign
        )
        frame = new if frame is None else pd.concat([frame, new], ignore_index=True)
        write_results(devices.output, frame, metadata)
        totals = new.loc[new["kind"] == "decision", "seconds"]
        progress.set_postfix_str(
            f"trial {index}: {len(decisions)} decisions, {totals.mean():.1f}s each, "
            f"{foreign:.1f} foreign cores",
            refresh=False,
        )
    print(
        f"{device} session finished: {len(work)} new trials, "
        f"{0 if frame is None else len(frame)} rows in {devices.output}.",
        flush=True,
    )
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.simulator_comparison.devices",
        description="One device's session of the CPU-against-GPU experiment: every "
        "rebalancing decision with JointFM and the solver on that device, timed "
        "by stage, resuming its results file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration of the CPU-against-GPU experiment.",
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        required=True,
        help="The session's device: GPU or CPU.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    devices, experiment = load_devices_config(args.config)
    select_session_gpus(args.device)
    run_device_session(
        devices,
        experiment,
        args.device,
        command=sys.argv if argv is None else ["main", *argv],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

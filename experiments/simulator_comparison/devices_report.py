# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tables and figures of the CPU-against-GPU experiment (``devices.py``).

``device_tables`` reads a results file into four tables, each holding exactly
the numbers its figure draws:

- ``runtime``: each simulator's mean seconds per scenario-generation call on
  every device it ran on.
- ``runtime_devices``: a rebalancing decision's stages per device (scenario
  generation, optimizer, and their total) and the CPU-to-GPU ratio of each.
- ``solvers``: each paired solver's mean seconds per solve of the decisions'
  identical problems, its time over the GPU pipeline's solver, and how far its
  solutions lie from that solver's.
- ``solver_sizes``: each paired solver's mean seconds per solve over the
  scenario counts of the sweep.

A mean is the mean over trials of each trial's mean, with its 95 % bootstrap
interval over trials (``analysis.bootstrap_interval``); a ratio is the median
over trials of each trial's ratio of means, with its interval. Only trials
that both sessions finished are read, and warm-up rows are left out of every
timing. Every table names the hardware: the host's CPU model, sockets, and
physical cores, and the GPU the GPU session ran on.

``write_device_figures`` writes the tables as CSV files and draws them in the
blog's style (``blog_style``). The blog writes them beside its other figures,
and this module's ``main`` writes them as the experiment's own report.
Titles state what is drawn, never a finding.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .analysis import BOOTSTRAP_SEED, bootstrap_interval
from .arms import ARMS_BY_NAME
from .blog_style import (
    BLOG_NAMES,
    TEXT_SIZE,
    _axis_labels,
    _footnote,
    _headline,
    _panel_title,
    _seconds_rows,
    _style_text,
)
from .config import load_devices_config
from .devices import DEVICES
from .plotting import (
    DEVICE_ORDER,
    INK,
    INK_SECONDARY,
    LINE_WIDTH,
    MARKER_EDGE,
    SIMULATOR_COLORS,
    SIMULATOR_ORDER,
    SURFACE,
    _curves,
    _figure,
    _save,
)
from .solvers import SOLVERS_BY_NAME
from .table import read_results

# Each stage column of a decision row, as the figures name it.
STAGES = {
    "generation_seconds": "Scenarios",
    "optimizer_seconds": "Optimizer",
    "seconds": "Decision",
}
# The reading of runtime_devices.csv that holds the CPU-to-GPU time ratio.
DEVICE_RATIO = "cpu_over_gpu"
# Slots of the reference palette (plotting.SIMULATOR_COLORS): JointFM's blue
# for its scenarios, the swap cells' amber for the optimizer, ink for the total.
STAGE_COLORS = {
    "generation_seconds": "#2a78d6",
    "optimizer_seconds": "#eda100",
    "seconds": INK,
}
# One reference-palette slot per solver, its cuOpt methods in the blue-violet
# range and the CPU solvers in the warm and green slots.
SOLVER_COLORS = {
    "CUOPT": "#2a78d6",
    "CUOPT_BARRIER": "#6250d6",
    "CUOPT_DUAL_SIMPLEX": "#e87ba4",
    "HIGHS": "#eb6834",
    "CLARABEL": "#1baf7a",
}
TABLES = ("runtime", "runtime_devices", "solvers", "solver_sizes")


@dataclass(frozen=True)
class SolverStyle:
    """How one solver is drawn: its own color, solid on the GPU and dashed on the CPU."""

    name: str

    @property
    def color(self) -> str:
        return SOLVER_COLORS[self.name]

    @property
    def filled(self) -> bool:
        return SOLVERS_BY_NAME[self.name].device == "GPU"

    @property
    def inline_label(self) -> str:
        solver = SOLVERS_BY_NAME[self.name]
        return f"{solver.label} on {solver.device}"


def read_device_results(path: Path) -> tuple[pd.DataFrame, dict]:
    """The rows of the trials both sessions finished, and the file's identity.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If either session is missing, or no trial is finished by both.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run task experiments:devices with a GPU and "
            f"a CPU service first."
        )
    frame, metadata = read_results(path)
    identity = metadata.identity
    sessions = {arm["device"] for arm in identity["arms"].values()}
    missing = [device for device in DEVICES if device not in sessions]
    if missing:
        raise ValueError(
            f"{path} holds no {missing} session; task experiments:devices runs both."
        )
    decisions = frame[frame["kind"] == "decision"]
    finished = set.intersection(
        *(set(decisions.loc[decisions["session"] == d, "trial"]) for d in DEVICES)
    )
    if not finished:
        raise ValueError(f"{path} holds no trial that both sessions finished.")
    return frame[frame["trial"].isin(finished)], identity


def hardware(identity: dict) -> dict[str, object]:
    """The host's CPU and the GPU session's GPU, as every table records them."""
    host = identity["host"]
    gpu = next(
        record["gpu"] for record in identity["services"].values() if "gpu" in record
    )
    return {
        "cpu": host["cpu"],
        "cpu_sockets": host["sockets"],
        "cpu_cores": host["physical_cores"],
        "gpu": gpu["name"],
    }


def _hardware_text(table: pd.DataFrame) -> str:
    first = table.iloc[0]
    sockets = int(first["cpu_sockets"])
    return (
        f"CPU: {f'{sockets} × ' if sockets > 1 else ''}{first['cpu']}, "
        f"{int(first['cpu_cores'])} cores in all. GPU: one {first['gpu']}."
    )


def _mean(values: pd.Series) -> dict:
    """Mean of per-trial values, with its bootstrap interval over trials."""
    array = values.dropna().to_numpy()
    low, high = bootstrap_interval(
        array, np.mean, np.random.default_rng(BOOTSTRAP_SEED)
    )
    return {
        "estimate": float(array.mean()),
        "ci_low": low,
        "ci_high": high,
        "trials": len(array),
    }


def _ratio(numerator: pd.Series, denominator: pd.Series) -> dict:
    """Median over trials of per-trial ratios, with its bootstrap interval."""
    ratios = (numerator / denominator).dropna().to_numpy()
    low, high = bootstrap_interval(
        ratios, np.median, np.random.default_rng(BOOTSTRAP_SEED)
    )
    return {
        "estimate": float(np.median(ratios)),
        "ci_low": low,
        "ci_high": high,
        "trials": len(ratios),
    }


def _timed(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    return frame[(frame["kind"] == kind) & ~frame["warmup"].astype(bool)]


def runtime_table(frame: pd.DataFrame, identity: dict) -> pd.DataFrame:
    """Each simulator's mean seconds per generation call, per device, over the decisions.

    ``num_scen`` is the median scenario count a call returned: the configured
    count, except for Historical, which replays the look-back window's days.
    """
    calls = _timed(frame, "generation")
    calls = calls[~calls["sweep"].astype(bool)]
    per_trial = calls.groupby(["name", "trial"])["seconds"].mean()

    def order(name: str) -> tuple[int, int]:
        arm = ARMS_BY_NAME[name]
        return SIMULATOR_ORDER.index(arm.fit_type), DEVICE_ORDER.index(arm.device)

    rows = [
        {
            "arm": name,
            "fit_type": ARMS_BY_NAME[name].fit_type,
            "device": ARMS_BY_NAME[name].device,
            **_mean(per_trial[name]),
            "num_scen": int(calls.loc[calls["name"] == name, "num_scen"].median()),
        }
        for name in sorted(calls["name"].unique(), key=order)
    ]
    return pd.DataFrame(rows).assign(
        assets=identity["experiment"]["assets"], **hardware(identity)
    )


def stage_table(frame: pd.DataFrame, identity: dict) -> pd.DataFrame:
    """Each stage of a pipeline decision per device, and its CPU-to-GPU ratio.

    Rows of reading ``seconds`` hold a device's mean seconds per decision of
    the stage; the row of reading ``DEVICE_RATIO`` holds the median over
    trials of the CPU's mean over the GPU's.
    """
    decisions = _timed(frame, "decision")
    per_trial = decisions.groupby(["session", "trial"])[list(STAGES)].mean()
    sessions = {arm["device"]: (name, arm) for name, arm in identity["arms"].items()}
    rows = []
    for stage in STAGES:
        for device in DEVICE_ORDER:
            name, arm = sessions[device]
            rows.append(
                {
                    "stage": stage,
                    "reading": "seconds",
                    "device": device,
                    "arm": name,
                    "solver": arm["solver"],
                    **_mean(per_trial.loc[device, stage]),
                }
            )
        rows.append(
            {
                "stage": stage,
                "reading": DEVICE_RATIO,
                "device": "",
                "arm": "",
                "solver": "",
                **_ratio(per_trial.loc["CPU", stage], per_trial.loc["GPU", stage]),
            }
        )
    return pd.DataFrame(rows).assign(
        num_scen=identity["devices"]["num_scen"],
        assets=identity["experiment"]["assets"],
        **hardware(identity),
    )


def solver_table(frame: pd.DataFrame, identity: dict) -> pd.DataFrame:
    """Each paired solver's seconds per solve, its time over the GPU solver's, and its distance.

    ``estimate`` is the mean seconds per solve and ``setup_seconds`` the mean
    seconds of the problem's setup; ``ratio`` is the median over trials of
    the solver's mean over the GPU pipeline solver's. ``max_weight_difference``
    is the largest absolute difference of an asset or cash weight from the GPU
    solver's solution of the same problem, and ``max_objective_gap`` the
    largest absolute difference of the objective, both over every decision.
    """
    solves = frame[(frame["kind"] == "solve") & ~frame["sweep"].astype(bool)]
    timed = solves[~solves["warmup"].astype(bool)]
    per_trial = timed.groupby(["name", "trial"])[["seconds", "setup_seconds"]].mean()
    reference_name = identity["devices"]["gpu_solver"]
    keys = ["trial", "decision"]
    reference = solves[solves["name"] == reference_name].set_index(keys)
    rows = []
    for name in identity["devices"]["paired_solvers"]:
        joined = (
            solves[solves["name"] == name]
            .set_index(keys)
            .join(reference, rsuffix="_reference", how="inner")
        )
        weights = np.abs(
            np.stack(joined["weights"].to_numpy())
            - np.stack(joined["weights_reference"].to_numpy())
        ).max(axis=1)
        cash = np.abs(joined["cash"] - joined["cash_reference"]).to_numpy()
        ratio = _ratio(
            per_trial.loc[name, "seconds"], per_trial.loc[reference_name, "seconds"]
        )
        rows.append(
            {
                "solver": name,
                "label": SOLVERS_BY_NAME[name].label,
                "device": SOLVERS_BY_NAME[name].device,
                **_mean(per_trial.loc[name, "seconds"]),
                "setup_seconds": float(per_trial.loc[name, "setup_seconds"].mean()),
                "ratio": ratio["estimate"],
                "ratio_ci_low": ratio["ci_low"],
                "ratio_ci_high": ratio["ci_high"],
                "max_weight_difference": float(np.maximum(weights, cash).max()),
                "max_objective_gap": float(
                    np.abs(joined["objective"] - joined["objective_reference"]).max()
                ),
                "decisions": len(joined),
            }
        )
    return pd.DataFrame(rows).assign(
        reference=reference_name,
        num_scen=identity["devices"]["num_scen"],
        assets=identity["experiment"]["assets"],
        **hardware(identity),
    )


def solver_size_table(frame: pd.DataFrame, identity: dict) -> pd.DataFrame:
    """Each paired solver's mean seconds per solve at every scenario count of the sweep.

    The sweep's decisions are each trial's first ``sweep_decisions``; at the
    configured ``num_scen`` their paired solves stand in for a sweep count.
    """
    solves = _timed(frame, "solve")
    solves = solves[solves["decision"] < identity["devices"]["sweep_decisions"]]
    per_trial = solves.groupby(["name", "num_scen", "trial"])["seconds"].mean()
    rows = [
        {
            "solver": name,
            "label": SOLVERS_BY_NAME[name].label,
            "device": SOLVERS_BY_NAME[name].device,
            "num_scen": int(count),
            **_mean(per_trial.loc[name, count]),
        }
        for name in identity["devices"]["paired_solvers"]
        for count in sorted(per_trial.loc[name].index.get_level_values(0).unique())
    ]
    return pd.DataFrame(rows).assign(
        assets=identity["experiment"]["assets"], **hardware(identity)
    )


def device_tables(results: Path) -> dict[str, pd.DataFrame]:
    """Every table of the module docstring, by name, from a results file.

    Raises
    ------
    FileNotFoundError, ValueError
        As ``read_device_results`` raises.
    """
    frame, identity = read_device_results(results)
    return {
        "runtime": runtime_table(frame, identity),
        "runtime_devices": stage_table(frame, identity),
        "solvers": solver_table(frame, identity),
        "solver_sizes": solver_size_table(frame, identity),
    }


def plot_runtime(table: pd.DataFrame, directory: Path) -> list[Path]:
    """Each simulator's mean seconds per scenario-generation call per device (``runtime_table``)."""
    figure, axes = _figure(8, 0.7 * len(table) + 2.8)
    ax = axes[0, 0]
    _seconds_rows(
        ax,
        [
            (
                f"{BLOG_NAMES[row.fit_type]} on {row.device}",
                SIMULATOR_COLORS[row.fit_type],
                row.estimate,
                row.ci_low,
                row.ci_high,
            )
            for row in table.itertuples()
        ],
        "call",
    )
    jointfm = table[table["fit_type"] == "jointfm"].iloc[0]
    _panel_title(
        ax,
        "Seconds per scenario-generation call",
        f"mean over {int(table['trials'].min())} trials with its 95 % interval;"
        " calls without warm-up",
    )
    _footnote(
        figure,
        f"A call draws {int(jointfm['num_scen']):,} next-day scenarios of "
        f"{int(jointfm['assets'])} assets from a rebalancing decision's look-back "
        "window\n(Historical replays the window's days); JointFM's call includes "
        f"the HTTP round trip to its local service.\n{_hardware_text(table)}",
    )
    _headline(figure, "Scenario generation on CPU and GPU: seconds per call")
    _style_text(figure)
    return _save(figure, directory, "runtime")


def plot_stages(table: pd.DataFrame, directory: Path) -> list[Path]:
    """A decision's stages per device, the GPU rows labeled with the CPU-to-GPU ratio (``stage_table``)."""
    figure, axes = _figure(8, 6.2)
    ax = axes[0, 0]
    seconds = table[table["reading"] == "seconds"].set_index(["stage", "device"])
    ratios = table[table["reading"] == DEVICE_RATIO].set_index("stage")
    rows = []
    for stage, stage_name in STAGES.items():
        for device in DEVICE_ORDER:
            row = seconds.loc[(stage, device)]
            label = f"{stage_name} on {device}"
            if stage == "optimizer_seconds":
                label = (
                    f"{stage_name} ({SOLVERS_BY_NAME[row['solver']].label}) on {device}"
                )
            if device == "GPU":
                ratio = ratios.loc[stage]
                label += (
                    f"\nCPU ÷ GPU {ratio['estimate']:.1f}× "
                    f"({ratio['ci_low']:.1f}–{ratio['ci_high']:.1f})"
                )
            rows.append(
                (
                    label,
                    STAGE_COLORS[stage],
                    row["estimate"],
                    row["ci_low"],
                    row["ci_high"],
                )
            )
    _seconds_rows(ax, rows, "decision")
    first = table.iloc[0]
    _panel_title(
        ax,
        "Seconds per rebalancing decision, by stage",
        f"mean over {int(seconds['trials'].min())} trials with its 95 % interval;"
        " decisions without warm-up",
    )
    _footnote(
        figure,
        f"A decision draws {int(first['num_scen']):,} JointFM scenarios of "
        f"{int(first['assets'])} assets and solves the Mean-CVaR problem on them;\n"
        "the scenarios include the HTTP round trip to JointFM's local service, the "
        "optimizer the problem's setup and solve.\nThe ratio is the median over "
        f"trials of CPU seconds over GPU seconds. {_hardware_text(table)}",
    )
    _headline(figure, "A rebalancing decision on CPU and GPU")
    _style_text(figure)
    return _save(figure, directory, "runtime_devices")


def plot_solvers(table: pd.DataFrame, directory: Path) -> list[Path]:
    """Each paired solver's mean seconds per solve, labeled with its time over the GPU solver's."""
    figure, axes = _figure(8, 0.75 * len(table) + 2.8)
    ax = axes[0, 0]
    reference = SOLVERS_BY_NAME[table["reference"].iloc[0]].label
    rows = []
    for row in table.itertuples():
        label = f"{row.label} on {row.device}"
        if row.solver != table["reference"].iloc[0]:
            label += (
                f"\n{row.ratio:.1f}× the time of {reference} "
                f"({row.ratio_ci_low:.1f}–{row.ratio_ci_high:.1f})"
            )
        rows.append(
            (label, SOLVER_COLORS[row.solver], row.estimate, row.ci_low, row.ci_high)
        )
    _seconds_rows(ax, rows, "solve")
    first = table.iloc[0]
    _panel_title(
        ax,
        "Seconds per solve of a rebalancing decision's problem",
        f"mean over {int(table['trials'].min())} trials with its 95 % interval;"
        " solves without warm-up, setup excluded",
    )
    _footnote(
        figure,
        f"Every solver solves the same {int(first['decisions'])} problems of "
        f"{int(first['num_scen']):,} scenarios of {int(first['assets'])} assets; "
        f"its weights differ from {reference}'s\nby at most "
        f"{table['max_weight_difference'].max():.1g}. The ratio is the median over "
        f"trials of the solver's seconds over {reference}'s.\n{_hardware_text(table)}",
    )
    _headline(figure, "Solving the Mean-CVaR problem on CPU and GPU")
    _style_text(figure)
    return _save(figure, directory, "solvers")


def plot_solver_sizes(table: pd.DataFrame, directory: Path) -> list[Path]:
    """Each paired solver's mean seconds per solve over the scenario counts (``solver_size_table``)."""
    figure, axes = _figure(8, 5.4)
    ax = axes[0, 0]
    styles = [SolverStyle(name) for name in table["solver"].unique()]
    _curves(ax, table.rename(columns={"solver": "arm"}), "num_scen", styles)
    ax.set_yscale("log")
    ax.set_xticks(ax.get_xticks(), [f"{int(tick):,}" for tick in ax.get_xticks()])
    _axis_labels(ax, "scenarios per problem (log)", "seconds per solve (log)")
    _panel_title(
        ax,
        "Seconds per solve over the number of scenarios",
        f"mean over {int(table['trials'].min())} or more trials with its 95 % "
        "interval; solves without warm-up, setup excluded",
    )
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=style.color,
                linewidth=LINE_WIDTH,
                linestyle="-" if style.filled else "--",
                marker="o",
                markerfacecolor=style.color if style.filled else SURFACE,
                markeredgecolor=style.color,
                markeredgewidth=MARKER_EDGE,
                label=style.inline_label,
            )
            for style in styles
        ],
        loc="outside lower center",
        ncols=min(len(styles), 3),
        frameon=False,
        fontsize=TEXT_SIZE,
        labelcolor=INK_SECONDARY,
    )
    _footnote(
        figure,
        f"Problems of {int(table['assets'].iloc[0])} assets from each trial's first "
        f"rebalancing decisions, with JointFM scenarios drawn on the GPU.\n"
        f"{_hardware_text(table)}",
    )
    _headline(figure, "Solve time over the number of scenarios")
    _style_text(figure)
    return _save(figure, directory, "solver_sizes")


def write_device_figures(
    tables: dict[str, pd.DataFrame], directory: Path
) -> list[Path]:
    """Write each table of ``device_tables`` as CSV and draw its figure into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for name in TABLES:
        path = directory / f"{name}.csv"
        tables[name].to_csv(path, index=False)
        files.append(path)
    files += plot_runtime(tables["runtime"], directory)
    files += plot_stages(tables["runtime_devices"], directory)
    files += plot_solvers(tables["solvers"], directory)
    files += plot_solver_sizes(tables["solver_sizes"], directory)
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.simulator_comparison.devices_report",
        description="Tables and figures of the CPU-against-GPU experiment, written "
        "to <output stem>_report beside its results; runs nothing.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration of the CPU-against-GPU experiment.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    devices, _ = load_devices_config(args.config)
    tables = device_tables(devices.output)
    files = write_device_figures(tables, devices.report_dir)
    columns = ["stage", "reading", "device", "solver", "estimate", "ci_low", "ci_high"]
    print(tables["runtime_devices"][columns].to_string(index=False))
    print(f"Device report finished: {len(files)} files in {devices.report_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

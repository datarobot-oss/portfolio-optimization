# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report of a simulator-comparison results file: tables and plots.

It reads only the results file, so it re-renders without rerunning any arm,
and answers which arm gives the most realistic scenarios on average (by the
energy score of the joint next-day distribution) and what each arm costs to
generate. Run it as ``python -m experiments.simulator_comparison.report``.

Colors follow the simulator, so the CPU and GPU arm of one simulator share a
hue; the device is the second channel: filled marks are CPU, hollow marks GPU,
and every axis label names the arm. The ``no_fit`` reference is gray.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

from .analysis import (
    COMPARED_SCORES,
    PIT_COLUMNS,
    REFERENCE_ARM,
    SKILL_SCORES,
    aggregate,
    comparisons,
    complete_trials,
    ranking,
    reliability,
    skill,
    skill_summary,
    summary,
    timing,
)
from .config import load_config
from .scores import PIT_BINS, ScenarioScores
from .table import read_results

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots 1-3 of the reference palette, validated all-pairs; the
# reference arm is gray because it is the baseline, not a contender.
SIMULATOR_COLORS = {
    "no_fit": MUTED,
    "gaussian": "#1baf7a",
    "kde": "#eb6834",
    "jointfm": "#2a78d6",
}
SIMULATOR_ORDER = ("no_fit", "gaussian", "kde", "jointfm")
DEVICE_ORDER = ("CPU", "GPU")
# (title, unit, target or None) per compared score. A score with a target is
# drawn as each arm's level against it; one without, where lower is better, as
# each arm's paired difference from the reference arm.
SCORE_LABELS = {
    "crps": ("CRPS per asset", "trailing-volatility units", None),
    "energy_score": (
        "Energy score of the asset vector",
        "trailing-volatility units",
        None,
    ),
    "spread": ("Spread", "scenario std over trailing volatility", 1.0),
    "cover_50": ("50 % interval coverage", "share of outcomes inside", 0.5),
    "cover_90": ("90 % interval coverage", "share of outcomes inside", 0.9),
    "corr_error_trailing": (
        "Correlation error, look-back window",
        "median absolute difference",
        None,
    ),
    "corr_bias_trailing": (
        "Correlation bias, look-back window",
        "median signed difference",
        0.0,
    ),
    "corr_error_forward": (
        "Correlation error, forward returns",
        "median absolute difference",
        None,
    ),
    "ew_var_exceeded": (
        "Equal-weight 95 % VaR exceedance",
        "share of origins",
        0.05,
    ),
}
SKILL_LABELS = {"energy_score": "Energy-score skill", "crps": "CRPS skill"}
FIGURE_DPI = 150
# Scatter area in points^2; a 6 pt (8 px at 96 dpi) marker.
MARKER_AREA = 36
MARKER_EDGE = 1.5
# The estimate's marker in a point-and-interval row: a 9 pt circle.
INTERVAL_MARKER_AREA = 81
INTERVAL_TICKS = 5
LINE_WIDTH = 2
JITTER = 0.12
# Fixed so the dot jitter is identical on every re-render.
JITTER_SEED = 0
BOX_WIDTH = 0.3
# Whiskers at the minimum and maximum, so a box draws no outlier dots.
BOX_WHISKERS = (0, 100)


def _duration(seconds: float) -> str:
    if seconds < 1:
        return f"{1000 * seconds:.3g} ms"
    return f"{seconds:.3g} s"


@dataclass(frozen=True)
class ArmStyle:
    """How one arm is drawn: hue by simulator, fill by device."""

    name: str
    fit_type: str
    device: str

    @property
    def color(self) -> str:
        return SIMULATOR_COLORS[self.fit_type]

    @property
    def filled(self) -> bool:
        return self.device == "CPU"

    @property
    def label(self) -> str:
        return f"{self.fit_type}\n{self.device}"

    @property
    def inline_label(self) -> str:
        return f"{self.fit_type} · {self.device}"


@dataclass(frozen=True)
class ReportOutputs:
    """Files a report wrote and the trials it left out."""

    files: list[Path]
    excluded_trials: list[int]


def arm_styles(per_trial: pd.DataFrame) -> list[ArmStyle]:
    """Arms in simulator order, CPU before GPU."""
    arms = per_trial.groupby("arm")[["fit_type", "device"]].first()
    styles = [ArmStyle(name, row.fit_type, row.device) for name, row in arms.iterrows()]

    def order(style: ArmStyle) -> tuple[int, int]:
        return (
            SIMULATOR_ORDER.index(style.fit_type),
            DEVICE_ORDER.index(style.device),
        )

    return sorted(styles, key=order)


def _figure(width: float, height: float, rows: int = 1, cols: int = 1):
    figure = Figure(figsize=(width, height), facecolor=SURFACE, layout="constrained")
    axes = figure.subplots(rows, cols, squeeze=False)
    for ax in axes.flat:
        _style_axes(ax)
    return figure, axes


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=AXIS, labelcolor=INK_SECONDARY, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _title(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=10, pad=14)
    ax.text(
        0,
        1.01,
        subtitle,
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8,
        va="bottom",
    )


def _dots(ax, x: float, values: np.ndarray, style: ArmStyle, rng, alpha=1.0) -> None:
    offsets = rng.uniform(-JITTER, JITTER, size=len(values))
    ax.scatter(
        x + offsets,
        values,
        s=MARKER_AREA,
        facecolor=style.color if style.filled else SURFACE,
        edgecolor=SURFACE if style.filled else style.color,
        linewidth=MARKER_EDGE,
        alpha=alpha,
        zorder=3,
    )


def _box(ax, x: float, values: np.ndarray, style: ArmStyle) -> None:
    line = {"color": style.color, "linewidth": 1}
    ax.boxplot(
        values,
        positions=[x],
        widths=BOX_WIDTH,
        whis=BOX_WHISKERS,
        patch_artist=True,
        boxprops={
            "facecolor": style.color if style.filled else SURFACE,
            "edgecolor": style.color,
            "linewidth": 1,
        },
        medianprops={**line, "color": SURFACE if style.filled else style.color},
        whiskerprops=line,
        capprops=line,
        manage_ticks=False,
        zorder=3,
    )


def _arm_axis(ax, styles: Sequence[ArmStyle]) -> None:
    ax.set_xticks(range(len(styles)), [style.label for style in styles])
    ax.set_xlim(-0.6, len(styles) - 0.4)


def _target_line(ax, target: float) -> None:
    ax.axhline(target, color=INK_SECONDARY, linewidth=1, zorder=2)


def _interval_panel(
    ax,
    rows: Sequence[tuple[ArmStyle, float, float, float]],
    *,
    reference: float,
    reference_label: str,
) -> None:
    """One row per arm: a point at its estimate and a line across its interval.

    ``rows`` holds (style, estimate, interval low, interval high), drawn top to
    bottom; a vertical line marks ``reference``, labeled at the bottom.
    """
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.axvline(reference, color=INK_SECONDARY, linewidth=1, zorder=2)
    ax.annotate(
        reference_label,
        xy=(reference, 0),
        xycoords=("data", "axes fraction"),
        xytext=(3, 3),
        textcoords="offset points",
        color=MUTED,
        fontsize=8,
    )
    for position, (style, estimate, low, high) in enumerate(rows):
        ax.hlines(position, low, high, color=style.color, linewidth=LINE_WIDTH)
        ax.scatter(
            [estimate],
            [position],
            s=INTERVAL_MARKER_AREA,
            facecolor=style.color if style.filled else SURFACE,
            edgecolor=style.color,
            linewidth=MARKER_EDGE,
            zorder=3,
        )
    ax.set_yticks(range(len(rows)), [row[0].inline_label for row in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    # Small differences carry long tick labels; more ticks would overlap.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=INTERVAL_TICKS))


def _device_legend(figure) -> None:
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=MUTED if device == "CPU" else SURFACE,
            markeredgecolor=SURFACE if device == "CPU" else MUTED,
            markeredgewidth=MARKER_EDGE,
            label=f"{device} ({'filled' if device == 'CPU' else 'hollow'})",
        )
        for device in DEVICE_ORDER
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncols=len(handles),
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
    )


def _save(figure: Figure, directory: Path, name: str) -> list[Path]:
    paths = [directory / f"{name}.png", directory / f"{name}.svg"]
    figure.savefig(paths[0], dpi=FIGURE_DPI, facecolor=SURFACE)
    figure.savefig(paths[1], facecolor=SURFACE)
    return paths


def plot_ranking(ranks: pd.DataFrame, styles: Sequence[ArmStyle], directory: Path):
    """Each arm's share of trials won by energy score, best mean rank on top."""
    by_name = {style.name: style for style in styles}
    order = list(ranks.index)
    figure, axes = _figure(8, 0.55 * len(order) + 1.6)
    ax = axes[0, 0]
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=1)
    for position, arm in enumerate(order):
        style = by_name[arm]
        share = ranks.loc[arm, "share_won"]
        ax.barh(
            position,
            share,
            height=0.5,
            color=style.color if style.filled else SURFACE,
            edgecolor=style.color,
            linewidth=LINE_WIDTH,
        )
        ax.text(
            share + 0.01,
            position,
            f"{share:.0%} won · mean rank {ranks.loc[arm, 'mean_rank']:.2f}",
            va="center",
            color=INK_SECONDARY,
            fontsize=8,
        )
    ax.set_yticks(range(len(order)), [by_name[arm].inline_label for arm in order])
    ax.invert_yaxis()
    # Room right of a full bar for its label; ticks stop at 100 %.
    ax.set_xlim(0, 1.3)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    _title(
        ax,
        "Most realistic scenarios",
        "share of trials in which the arm had the lowest energy score",
    )
    _device_legend(figure)
    return _save(figure, directory, "ranking")


def plot_scores(
    table: pd.DataFrame,
    levels: pd.DataFrame,
    styles: Sequence[ArmStyle],
    directory: Path,
):
    """One panel per score, each arm as a point with its 95 % interval.

    A score where lower is better shows each arm's median paired difference
    from the reference arm (``comparisons``), which removes the spread between
    trials that every arm shares. A score with a target shows each arm's mean
    over trials (``summary``) against that target.
    """
    contenders = [style for style in styles if style.name != REFERENCE_ARM]
    differences = table[table["reference"] == REFERENCE_ARM]
    figure, axes = _figure(13, 9, rows=3, cols=3)
    for ax, column in zip(axes.flat, COMPARED_SCORES):
        title, unit, target = SCORE_LABELS[column]
        if target is None:
            rows = differences[differences["metric"] == column].set_index("arm")
            _interval_panel(
                ax,
                [
                    (
                        style,
                        *rows.loc[
                            style.name, ["median_difference", "ci_low", "ci_high"]
                        ],
                    )
                    for style in contenders
                ],
                reference=0.0,
                reference_label=REFERENCE_ARM,
            )
            subtitle = f"minus {REFERENCE_ARM}, paired by trial; left is better"
        else:
            rows = levels[levels["metric"] == column].set_index("arm")
            _interval_panel(
                ax,
                [
                    (style, *rows.loc[style.name, ["mean", "ci_low", "ci_high"]])
                    for style in styles
                ],
                reference=target,
                reference_label=f"target {target:g}",
            )
            subtitle = "mean over trials; closest to the target is best"
        ax.set_xlabel(unit, color=INK_SECONDARY, fontsize=8)
        _title(ax, title, subtitle)
    figure.suptitle(
        "Scenario scores: point and 95 % bootstrap interval over trials",
        color=INK,
        fontsize=10,
    )
    _device_legend(figure)
    return _save(figure, directory, "scores")


def plot_skill(
    skills: pd.DataFrame,
    medians: pd.DataFrame,
    styles: Sequence[ArmStyle],
    directory: Path,
):
    """Per-trial skill against the reference, with the median and its interval."""
    rng = np.random.default_rng(JITTER_SEED)
    contenders = [style for style in styles if style.name != REFERENCE_ARM]
    figure, axes = _figure(12, 5, cols=len(SKILL_SCORES))
    for ax, score in zip(axes.flat, SKILL_SCORES):
        ax.axhline(0, color=INK_SECONDARY, linewidth=1, zorder=2)
        for position, style in enumerate(contenders):
            values = skills.loc[
                (skills["score"] == score) & (skills["arm"] == style.name), "skill"
            ].to_numpy()
            _dots(ax, position, values, style, rng)
            row = medians.loc[
                (medians["score"] == score) & (medians["arm"] == style.name)
            ].iloc[0]
            ax.vlines(
                position + 0.3,
                row["ci_low"],
                row["ci_high"],
                color=INK_SECONDARY,
                linewidth=LINE_WIDTH,
            )
            ax.hlines(
                row["median_skill"],
                position + 0.22,
                position + 0.38,
                color=INK,
                linewidth=LINE_WIDTH,
            )
        _arm_axis(ax, contenders)
        ax.set_ylabel(f"% better than {REFERENCE_ARM}", color=INK_SECONDARY, fontsize=8)
        _title(
            ax,
            SKILL_LABELS[score],
            "per trial; bar: median and 95 % interval; above 0 beats the reference",
        )
    _device_legend(figure)
    return _save(figure, directory, "skill")


def plot_pit(per_trial: pd.DataFrame, styles: Sequence[ArmStyle], directory: Path):
    """Each arm's mean PIT histogram against the flat line of a calibrated forecast."""
    pit = per_trial[list(PIT_COLUMNS)].groupby("arm").mean()
    cols = 3
    rows = -(-len(styles) // cols)
    figure, axes = _figure(12, 3.2 * rows, rows=rows, cols=cols)
    centers = (np.arange(PIT_BINS) + 0.5) / PIT_BINS
    for ax, style in zip(axes.flat, styles):
        ax.bar(
            centers,
            pit.loc[style.name].to_numpy(),
            width=0.8 / PIT_BINS,
            color=style.color if style.filled else SURFACE,
            edgecolor=style.color,
            linewidth=1.5,
        )
        _target_line(ax, 1 / PIT_BINS)
        ax.set_xlim(0, 1)
        _title(ax, style.inline_label, "share of assets per PIT bin")
    for ax in list(axes.flat)[len(styles) :]:
        ax.set_visible(False)
    figure.suptitle(
        "Calibration: flat at 1/20 is calibrated; U-shaped is too narrow, "
        "hump-shaped too wide",
        color=INK,
        fontsize=10,
    )
    _device_legend(figure)
    return _save(figure, directory, "calibration_pit")


def plot_reliability(levels: pd.DataFrame, styles: Sequence[ArmStyle], directory):
    """Empirical against nominal coverage of central intervals, per arm."""
    figure, axes = _figure(6.5, 6.5)
    ax = axes[0, 0]
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.plot([0, 1], [0, 1], color=INK_SECONDARY, linewidth=1, zorder=2)
    for style in styles:
        ax.plot(
            levels.columns,
            levels.loc[style.name].to_numpy(),
            color=style.color,
            linewidth=LINE_WIDTH,
            linestyle="-" if style.filled else "--",
            marker="o",
            markersize=7,
            markerfacecolor=style.color if style.filled else SURFACE,
            markeredgecolor=SURFACE if style.filled else style.color,
            markeredgewidth=MARKER_EDGE,
            label=style.inline_label,
            zorder=3,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal coverage", color=INK_SECONDARY, fontsize=8)
    ax.set_ylabel("empirical coverage", color=INK_SECONDARY, fontsize=8)
    _title(ax, "Interval reliability", "below the diagonal: intervals too narrow")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper left")
    return _save(figure, directory, "calibration_reliability")


def plot_timing(
    frame: pd.DataFrame,
    means: pd.DataFrame,
    speedups: pd.DataFrame,
    styles: Sequence[ArmStyle],
    directory: Path,
):
    """Seconds per scenario-generation call per arm, with GPU speedups."""
    timed = frame[~frame["warmup"]]
    figure, axes = _figure(12, 6)
    ax = axes[0, 0]
    positions = {style.name: position for position, style in enumerate(styles)}
    for position, style in enumerate(styles):
        seconds = timed.loc[timed["arm"] == style.name, "seconds"].to_numpy()
        _box(ax, position, seconds, style)
        row = means.loc[style.name]
        ax.vlines(
            position + 0.3,
            row["ci_low"],
            row["ci_high"],
            color=INK_SECONDARY,
            linewidth=LINE_WIDTH,
        )
        ax.hlines(
            row["mean_seconds"],
            position + 0.22,
            position + 0.38,
            color=INK,
            linewidth=LINE_WIDTH,
        )
        ax.text(
            position + 0.42,
            row["mean_seconds"],
            _duration(row["mean_seconds"]),
            va="center",
            color=INK_SECONDARY,
            fontsize=8,
        )
    ax.set_yscale("log")
    for row in speedups.itertuples():
        highest = timed.loc[timed["arm"] == row.gpu_arm, "seconds"].max()
        ax.text(
            positions[row.gpu_arm],
            highest * 1.5,
            f"CPU ÷ GPU {row.median_speedup:.1f}×\n"
            f"95 % CI {row.ci_low:.1f}–{row.ci_high:.1f}",
            ha="center",
            va="bottom",
            color=INK_SECONDARY,
            fontsize=8,
        )
    ax.set_ylim(top=timed["seconds"].max() * 6)
    _arm_axis(ax, styles)
    ax.set_ylabel("seconds per call (log)", color=INK_SECONDARY, fontsize=8)
    _title(
        ax,
        "Scenario-generation time",
        "calls without warm-up; box: median, quartiles, and range;"
        " bar: mean and 95 % interval over trials",
    )
    _device_legend(figure)
    return _save(figure, directory, "timing")


def _check_scores_recorded(identity: dict) -> None:
    recorded = [m["columns"] for m in identity["measurements"]]
    if list(ScenarioScores.columns) not in recorded:
        raise ValueError(
            "The results file holds no ScenarioScores columns, so it cannot be "
            "reported."
        )


def write_report(
    results_path: Path, output_dir: Path, exclude_arms: Sequence[str] = ()
) -> ReportOutputs:
    """Compute the statistics of a results file and write its tables and plots.

    ``exclude_arms`` leaves arms out of the report, for example an arm whose
    session is still running: a trial counts as complete when every reported
    arm has every origin, so an unfinished arm would otherwise hold back all
    trials it has not reached.

    Raises
    ------
    ValueError
        If the file holds no scenario scores, an excluded arm is not in the
        file, the reference arm is missing or excluded, or fewer trials are
        complete than the report needs.
    """
    frame, metadata = read_results(results_path)
    identity = metadata.identity
    _check_scores_recorded(identity)
    unknown = sorted(set(exclude_arms) - set(identity["arms"]))
    if unknown:
        raise ValueError(
            f"Cannot exclude {unknown}: the results hold {list(identity['arms'])}."
        )
    arms = tuple(arm for arm in identity["arms"] if arm not in exclude_arms)
    frame = frame[frame["arm"].isin(arms)]
    frame, excluded = complete_trials(frame, arms, identity["experiment"]["origins"])
    per_trial = aggregate(frame)
    styles = arm_styles(per_trial)
    skills = skill(per_trial)
    skill_medians = skill_summary(skills)
    ranks = ranking(per_trial)
    means, speedups = timing(per_trial)
    levels = reliability(per_trial)
    tables = {
        "summary": summary(per_trial),
        "comparisons": comparisons(per_trial),
        "skill": skill_medians,
        "ranking": ranks.rename_axis("arm").reset_index(),
        "timing": means.reset_index(),
        "speedup": speedups,
        "reliability": levels.rename_axis("arm").reset_index(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        files.append(path)
    files += plot_ranking(ranks, styles, output_dir)
    files += plot_scores(tables["comparisons"], tables["summary"], styles, output_dir)
    files += plot_skill(skills, skill_medians, styles, output_dir)
    files += plot_pit(per_trial, styles, output_dir)
    files += plot_reliability(levels, styles, output_dir)
    files += plot_timing(frame, means, speedups, styles, output_dir)
    return ReportOutputs(files=files, excluded_trials=excluded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.simulator_comparison.report",
        description="Tables and plots of the results of a simulator-comparison "
        "configuration, written to <output stem>_report beside its results file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration file the results were produced with.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="ARM",
        help="Arms to leave out, such as one whose session is still running.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output_dir = config.report_dir
    outputs = write_report(config.output, output_dir, exclude_arms=args.exclude)
    if args.exclude:
        print(f"Left out arms: {args.exclude}")
    if outputs.excluded_trials:
        print(
            f"Left out {len(outputs.excluded_trials)} incomplete trials: "
            f"{outputs.excluded_trials}"
        )
    with pd.option_context("display.width", 160, "display.precision", 3):
        for name in ("ranking", "skill", "speedup"):
            print(f"\n{name}:\n{pd.read_csv(output_dir / f'{name}.csv').to_string()}")
    print(
        f"\nSimulator comparison report finished: {len(outputs.files)} files in "
        f"{output_dir}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report of a simulator-comparison results file: tables and plots.

It reads the results file and the price file they were measured on, so it
re-renders without rerunning any arm. It answers which arm gives the most
realistic scenarios on average (by the energy score of the joint next-day
distribution and by two scores that react to wrong dependence, the variogram
score and the portfolio CRPS), how the portfolio readings change with the
portfolio size (``portfolio_analysis``), how every reading changes across
market regimes of the origins (``regimes``) and over the horizons after the
origin, and what each arm costs to generate. With the dependence swap arms
(``swaps``) it also splits every reading into what JointFM's marginals and
its dependence cause (``swap_effects.csv``). Every table holds every horizon
in its ``horizon`` column; the figures other than the horizon figures show
the next day. Run it as
``python -m experiments.simulator_comparison.report``. The figures follow
the shared look of ``plotting``.
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter

from .analysis import (
    COMPARED_SCORES,
    PIT_COLUMNS,
    RANKING_SCORES,
    REFERENCE_ARM,
    SKILL_SCORES,
    aggregate,
    comparisons,
    complete_trials,
    ranking,
    reliability,
    skill,
    skill_summary,
    stack_horizons,
    summary,
    swap_design,
    swap_effects,
    timing,
)
from .config import load_config
from .plotting import (
    GRID,
    INK,
    INK_SECONDARY,
    INTERVAL_TICKS,
    JITTER_SEED,
    LINE_WIDTH,
    MARKER_AREA,
    MARKER_EDGE,
    SURFACE,
    ArmStyle,
    _arm_axis,
    _arm_legend,
    _box,
    _curves,
    _device_legend,
    _dots,
    _figure,
    _interval_panel,
    _save,
    _target_line,
    _title,
    arm_styles,
)
from .portfolio_analysis import portfolio_pit, size_curves
from .portfolio_report import (
    plot_portfolio_horizons,
    plot_portfolio_pit,
    plot_portfolio_sizes,
)
from .regimes import (
    FEATURES,
    OVERALL,
    REGIMES,
    check_prices,
    origin_features,
    regime_readings,
    terciles,
)
from .runner import load_prices
from .scores import (
    PIT_BINS,
    PORTFOLIO_CRPS,
    PORTFOLIO_READINGS,
    TRAILING_STD_WINDOW,
    VARIOGRAM_ORDER,
    ScenarioScores,
    cell_column,
    recorded_cells,
)
from .table import read_results

# (title, unit, target or None) per compared score. A score with a target is
# drawn as each arm's level against it; one without, where lower is better, as
# each arm's paired difference from the reference arm.
SCORE_LABELS = {
    "crps": ("CRPS per asset", "trailing volatility × √h units", None),
    "energy_score": (
        "Energy score of the asset vector",
        "trailing volatility × √h units",
        None,
    ),
    "variogram_score": (
        f"Variogram score of order {VARIOGRAM_ORDER}",
        "trailing volatility × √h units",
        None,
    ),
    "spread": ("Spread", "scenario std over trailing volatility × √h", 1.0),
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
    "mean_correlation": (
        "Expected-return correlation with realized returns",
        "correlation across assets; 0 carries no information",
        0.0,
    ),
    PORTFOLIO_CRPS: (
        "Portfolio CRPS, mean over portfolios",
        "portfolio trailing volatility × √h units",
        None,
    ),
}
SKILL_LABELS = {
    "energy_score": "Energy-score skill",
    "variogram_score": "Variogram-score skill",
    PORTFOLIO_CRPS: "Portfolio-CRPS skill",
    "crps": "CRPS skill",
}
FEATURE_LABELS = {
    "window_correlation": "look-back window's mean pairwise correlation",
    "volatility_ratio": f"volatility ratio, last {TRAILING_STD_WINDOW} returns "
    "over the whole look-back window",
    "correlation_change": "change of the mean pairwise correlation over the "
    "forward returns (looks ahead)",
}
# Share of a regime column that one regime's arms spread across.
REGIME_GROUP_WIDTH = 0.7
PANEL_COLUMNS = 3


def _duration(seconds: float) -> str:
    if seconds < 1:
        return f"{1000 * seconds:.3g} ms"
    return f"{seconds:.3g} s"


@dataclass(frozen=True)
class ReportOutputs:
    """Files a report wrote and the trials it left out."""

    files: list[Path]
    excluded_trials: list[int]


def plot_ranking(ranks: pd.DataFrame, styles: Sequence[ArmStyle], directory: Path):
    """Each arm's share of trials won, one panel per ranking score side by side.

    Every panel lists the arms by that score's mean rank, best on top, so an
    arm that wins on the energy score but loses on a dependence-aware score
    stands out across the row.
    """
    by_name = {style.name: style for style in styles}
    figure, axes = _figure(
        6 * len(RANKING_SCORES), 0.55 * len(styles) + 1.6, cols=len(RANKING_SCORES)
    )
    for ax, score in zip(axes.flat, RANKING_SCORES):
        rows = ranks[ranks["score"] == score].set_index("arm")
        order = list(rows.index)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=1)
        for position, arm in enumerate(order):
            style = by_name[arm]
            share = rows.loc[arm, "share_won"]
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
                f"{share:.0%} won · mean rank {rows.loc[arm, 'mean_rank']:.2f}",
                va="center",
                color=INK_SECONDARY,
                fontsize=8,
            )
        ax.set_yticks(range(len(order)), [by_name[arm].inline_label for arm in order])
        ax.invert_yaxis()
        # Room right of a full bar for its label; ticks stop at 100 %.
        ax.set_xlim(0, 1.45)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        _title(
            ax,
            SCORE_LABELS[score][0],
            "share of trials in which the arm had the lowest score",
        )
    figure.suptitle(
        "Most realistic scenarios, by each proper score on its own",
        color=INK,
        fontsize=10,
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
    rows = -(-len(COMPARED_SCORES) // PANEL_COLUMNS)
    figure, axes = _figure(13, 3 * rows, rows=rows, cols=PANEL_COLUMNS)
    for ax in list(axes.flat)[len(COMPARED_SCORES) :]:
        ax.set_visible(False)
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
    figure, axes = _figure(5 * len(SKILL_SCORES), 5, cols=len(SKILL_SCORES))
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


def _regime_panel(
    ax,
    columns: Sequence[tuple[str, str]],
    labels: Sequence[str],
    arms: Sequence[ArmStyle],
    values: dict[tuple[str, str, str], tuple[float, float, float]],
    *,
    reference: float,
) -> None:
    """One column per (feature, regime), each arm as a point with its interval.

    ``values`` maps (feature, regime, arm) to (estimate, interval low, interval
    high); a horizontal line marks ``reference``, and a thin line separates the
    first column, the readings over all origins, from the regimes.
    """
    ax.axhline(reference, color=INK_SECONDARY, linewidth=1, zorder=2)
    ax.axvline(0.5, color=GRID, linewidth=1, zorder=1)
    offsets = (np.arange(len(arms)) - (len(arms) - 1) / 2) * (
        REGIME_GROUP_WIDTH / len(arms)
    )
    for position, column in enumerate(columns):
        for offset, style in zip(offsets, arms):
            estimate, low, high = values[(*column, style.name)]
            ax.vlines(position + offset, low, high, color=style.color, linewidth=1)
            ax.scatter(
                [position + offset],
                [estimate],
                s=MARKER_AREA,
                facecolor=style.color if style.filled else SURFACE,
                edgecolor=style.color,
                linewidth=MARKER_EDGE,
                zorder=3,
            )
    ax.set_xticks(range(len(columns)), labels)
    ax.set_xlim(-0.6, len(columns) - 0.4)


def _cells(
    table: pd.DataFrame, columns: Sequence[str], keys: Sequence[str]
) -> dict[tuple, tuple[float, float, float]]:
    keyed = table.set_index([*keys, "arm"])[list(columns)]
    return {key: tuple(row) for key, row in zip(keyed.index, keyed.to_numpy())}


def _reading_panels(
    readings: dict[str, pd.DataFrame], styles: Sequence[ArmStyle], keys: Sequence[str]
) -> list[tuple]:
    """One panel per compared score and skill, drawn as in ``plot_scores`` and
    ``plot_skill``.

    ``readings`` holds the ``summary``, ``comparisons``, and ``skill`` tables
    with the columns ``keys`` that name each panel position, such as the
    feature and regime, or the horizon. A score where lower is better is each
    arm's median paired difference from the reference arm, a score with a
    target each arm's mean over trials against the target, and a skill each
    arm's median skill.

    Returns
    -------
    list of tuple
        (title, unit, arms, values, reference) per panel, where ``values``
        maps (*key values, arm) to (estimate, interval low, interval high).
    """
    contenders = [style for style in styles if style.name != REFERENCE_ARM]
    differences = readings["comparisons"][
        readings["comparisons"]["reference"] == REFERENCE_ARM
    ]
    levels = readings["summary"]
    skills = readings["skill"]
    panels = []
    for column in COMPARED_SCORES:
        title, unit, target = SCORE_LABELS[column]
        if target is None:
            panels.append(
                (
                    title,
                    f"{unit}, minus {REFERENCE_ARM}; lower is better",
                    contenders,
                    _cells(
                        differences[differences["metric"] == column],
                        ["median_difference", "ci_low", "ci_high"],
                        keys,
                    ),
                    0.0,
                )
            )
        else:
            panels.append(
                (
                    title,
                    f"{unit}, mean over trials; target {target:g}",
                    styles,
                    _cells(
                        levels[levels["metric"] == column],
                        ["mean", "ci_low", "ci_high"],
                        keys,
                    ),
                    target,
                )
            )
    for score in SKILL_SCORES:
        panels.append(
            (
                SKILL_LABELS[score],
                f"% better than {REFERENCE_ARM}, median over trials",
                contenders,
                _cells(
                    skills[skills["score"] == score],
                    ["median_skill", "ci_low", "ci_high"],
                    keys,
                ),
                0.0,
            )
        )
    return panels


def plot_regimes(
    readings: dict[str, pd.DataFrame],
    bounds: pd.DataFrame,
    styles: Sequence[ArmStyle],
    directory: Path,
) -> list[Path]:
    """Per origin feature, every reading in each tercile beside all origins.

    One figure per feature, one panel per score and skill, drawn as in
    ``plot_scores`` and ``plot_skill``: a score where lower is better as each
    arm's median paired difference from the reference arm, a score with a
    target as each arm's mean over trials against the target, and a skill as
    each arm's median skill. Each panel's first column holds the readings over
    all origins, the next three the feature's terciles, low to high.
    """
    panels = _reading_panels(readings, styles, ["feature", "regime"])
    files = []
    for feature in FEATURES:
        rows = bounds[bounds["feature"] == feature].set_index("regime")
        columns = [(OVERALL, OVERALL), *((feature, name) for name in REGIMES)]
        labels = ["all origins"] + [
            f"{name}\n{rows.loc[name, 'low']:.2f} to {rows.loc[name, 'high']:.2f}"
            for name in REGIMES
        ]
        rows = -(-len(panels) // PANEL_COLUMNS)
        figure, axes = _figure(14, 3.75 * rows, rows=rows, cols=PANEL_COLUMNS)
        for ax, (title, unit, arms, values, reference) in zip(axes.flat, panels):
            _regime_panel(ax, columns, labels, arms, values, reference=reference)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=INTERVAL_TICKS))
            _title(ax, title, unit)
        for ax in list(axes.flat)[len(panels) :]:
            ax.set_visible(False)
        figure.suptitle(
            f"Readings per tercile of the {FEATURE_LABELS[feature]}: point and 95 % "
            f"bootstrap interval over trials",
            color=INK,
            fontsize=10,
        )
        _arm_legend(figure, styles)
        files += _save(figure, directory, f"regimes_{feature}")
    return files


def plot_horizons(
    readings: dict[str, pd.DataFrame], styles: Sequence[ArmStyle], directory: Path
) -> list[Path]:
    """Every score and skill over the horizon, one panel each.

    ``readings`` holds the ``summary``, ``comparisons``, and ``skill`` tables
    stacked over the horizons (``stack_horizons``); the panels are those of
    ``plot_regimes`` (``_reading_panels``), each arm a line through its
    estimates at the horizons, with its interval at each.
    """
    panels = _reading_panels(readings, styles, ["horizon"])
    horizons = sorted(readings["summary"]["horizon"].unique())
    rows = -(-len(panels) // PANEL_COLUMNS)
    figure, axes = _figure(14, 3.6 * rows, rows=rows, cols=PANEL_COLUMNS)
    for ax, (title, unit, arms, values, reference) in zip(axes.flat, panels):
        table = pd.DataFrame(
            [
                (horizon, style.name, *values[(horizon, style.name)])
                for horizon in horizons
                for style in arms
            ],
            columns=["horizon", "arm", "estimate", "ci_low", "ci_high"],
        )
        _target_line(ax, reference)
        _curves(ax, table, "horizon", arms)
        ax.set_xlabel("trading days after the origin", fontsize=8)
        _title(ax, title, unit)
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    figure.suptitle(
        "Readings over the horizon after the origin: point and 95 % bootstrap "
        "interval over trials",
        color=INK,
        fontsize=10,
    )
    _arm_legend(figure, styles)
    return _save(figure, directory, "horizons")


def _check_scores_recorded(identity: dict) -> None:
    recorded = [m["columns"] for m in identity["measurements"]]
    if list(ScenarioScores.columns) not in recorded:
        raise ValueError(
            "The results file holds no ScenarioScores columns, so it cannot be "
            "reported."
        )
    if not any(PORTFOLIO_CRPS in columns for columns in recorded):
        raise ValueError(
            "The results file holds no PortfolioScores columns, so it cannot be "
            "reported."
        )
    if "horizons" not in identity["experiment"]:
        raise ValueError(
            "The results file records no horizons, so it cannot be reported; "
            "rerun it from scratch."
        )
    if "origin_history" not in identity["experiment"]:
        raise ValueError(
            "The results file records no origin_history, so its trials cannot be "
            "drawn again; rerun it from scratch."
        )


def write_report(
    results_path: Path,
    prices_path: Path,
    output_dir: Path,
    exclude_arms: Sequence[str] = (),
) -> ReportOutputs:
    """Compute the statistics of a results file and write its tables and plots.

    ``prices_path`` is the price file the results were measured on; the
    regime readings recompute each origin's features from it. ``exclude_arms``
    leaves arms out of the report, for example an arm whose session is still
    running: a trial counts as complete when every reported arm has every
    origin at every horizon, so an unfinished arm would otherwise hold back
    all trials it has not reached. Every statistic is computed per horizon;
    the tables stack them, and the figures other than the horizon figures
    show the next day.

    Raises
    ------
    ValueError
        If the file holds no scenario or portfolio scores or no horizons, the
        price file is not the one the results were measured on, an excluded arm is not in the file, the
        reference arm is missing or excluded, fewer trials are complete than
        the report needs, a regime holds too few trials or has tied bounds, or
        the swap design is incomplete.
    """
    frame, metadata = read_results(results_path)
    identity = metadata.identity
    _check_scores_recorded(identity)
    check_prices(prices_path, identity)
    unknown = sorted(set(exclude_arms) - set(identity["arms"]))
    if unknown:
        raise ValueError(
            f"Cannot exclude {unknown}: the results hold {list(identity['arms'])}."
        )
    arms = tuple(arm for arm in identity["arms"] if arm not in exclude_arms)
    frame = frame[frame["arm"].isin(arms)]
    experiment = identity["experiment"]
    horizons = experiment["horizons"]
    frame, excluded = complete_trials(frame, arms, experiment["origins"], len(horizons))
    frames = {horizon: frame[frame["horizon"] == horizon] for horizon in horizons}
    per_trial = {horizon: aggregate(rows) for horizon, rows in frames.items()}
    next_day = per_trial[1]
    styles = arm_styles(next_day)
    skills = {horizon: skill(table) for horizon, table in per_trial.items()}
    readings = {
        "summary": {h: summary(table) for h, table in per_trial.items()},
        "comparisons": {h: comparisons(table) for h, table in per_trial.items()},
        "skill": {h: skill_summary(table) for h, table in skills.items()},
        "ranking": {h: ranking(table) for h, table in per_trial.items()},
        "portfolio_sizes": {h: size_curves(table) for h, table in per_trial.items()},
        "portfolio_pit": {h: portfolio_pit(table) for h, table in per_trial.items()},
        "reliability": {
            h: reliability(table).rename_axis("arm").reset_index()
            for h, table in per_trial.items()
        },
    }
    means, speedups = timing(next_day)
    features = origin_features(load_prices(prices_path), frame, experiment)
    regimes, bounds = terciles(features)
    regime_tables = {
        horizon: regime_readings(rows, regimes) for horizon, rows in frames.items()
    }
    design = swap_design({arm: identity["arms"][arm] for arm in arms})
    if design is not None:
        effect_columns = [
            *COMPARED_SCORES,
            *(
                cell_column(kind, size, reading)
                for kind, size in recorded_cells(frame.columns)
                for reading in PORTFOLIO_READINGS
                if not reading.startswith("pit_")
            ),
        ]
        readings["swap_effects"] = {
            h: swap_effects(table, effect_columns, design)
            for h, table in per_trial.items()
        }
    stacked = {name: stack_horizons(tables) for name, tables in readings.items()}
    tables = {
        **stacked,
        "timing": means.reset_index(),
        "speedup": speedups,
        "origin_features": features.join(regimes, rsuffix="_regime").reset_index(),
        "regimes": bounds,
        **{
            f"regime_{name}": stack_horizons(
                {h: by_name[name] for h, by_name in regime_tables.items()}
            )
            for name in regime_tables[1]
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        files.append(path)
    files += plot_ranking(readings["ranking"][1], styles, output_dir)
    files += plot_scores(
        readings["comparisons"][1], readings["summary"][1], styles, output_dir
    )
    files += plot_skill(skills[1], readings["skill"][1], styles, output_dir)
    files += plot_pit(next_day, styles, output_dir)
    files += plot_reliability(reliability(next_day), styles, output_dir)
    files += plot_timing(frames[1], means, speedups, styles, output_dir)
    files += plot_regimes(regime_tables[1], bounds, styles, output_dir)
    files += plot_portfolio_sizes(readings["portfolio_sizes"][1], styles, output_dir)
    files += plot_portfolio_pit(readings["portfolio_pit"][1], styles, output_dir)
    files += plot_horizons(stacked, styles, output_dir)
    files += plot_portfolio_horizons(stacked["portfolio_sizes"], styles, output_dir)
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
    outputs = write_report(
        config.output, config.prices, output_dir, exclude_arms=args.exclude
    )
    if args.exclude:
        print(f"Left out arms: {args.exclude}")
    if outputs.excluded_trials:
        print(
            f"Left out {len(outputs.excluded_trials)} incomplete trials: "
            f"{outputs.excluded_trials}"
        )
    with pd.option_context("display.width", 160, "display.precision", 3):
        for name in ("ranking", "skill", "speedup", "regimes"):
            print(f"\n{name}:\n{pd.read_csv(output_dir / f'{name}.csv').to_string()}")
    print(
        f"\nSimulator comparison report finished: {len(outputs.files)} files in "
        f"{output_dir}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

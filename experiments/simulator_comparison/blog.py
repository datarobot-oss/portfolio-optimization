# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blog figures of a configuration's results: the readings a post about JointFM shows.

The two reports (``report``, ``rebalancing_report``) draw every reading of
every arm, so they show where JointFM needs to improve as much as where it
leads. This module draws only the readings the blog post builds its story on,
for the scenario simulators, from the scenario and rebalancing results one
configuration names, into ``<output stem>_blog/`` beside the scenario results
(``ExperimentConfig.blog_dir``). Each figure is a PNG and an SVG, with a CSV
of exactly the numbers it draws.

- ``ranking``: per score of ``BLOG_RANKING_SCORES``, each simulator's share of
  trials with the best next-day score.
- ``skill_horizons``: skill against replaying the window per score of
  ``BLOG_SKILL_SCORES``, over the horizons.
- ``volatility_regimes``: next-day spread and 90 % interval coverage over all
  origins and per tercile of the volatility ratio (``regimes``).
- ``growth``: the rebalanced portfolio's mean cumulative return and mean
  drawdown from its running peak after every step.
- ``risk_return``: each simulator's mean annualized volatility against its mean
  annualized return, cash at the backtests' mean risk-free return, and the
  frontier with cash: the line from cash through the simulator whose line from
  cash is steepest (``risk_return_table``).
- ``performance``: every reading of ``PERFORMANCE_METRICS``, with the number of
  trials in which each simulator did better than replaying the window.

Every statistic comes from ``analysis``, ``regimes``, and
``rebalancing_report``, so the numbers equal the reports'. The figures name the
simulators as the post does (``BLOG_NAMES``) and draw one arm per simulator,
so each name appears once: its GPU arm where it ran on both devices. Titles
describe what is drawn and never state a finding, because the same figures
re-render from every run.
"""

import argparse
import hashlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from portfolio_optimization import risk_free

from .analysis import (
    REFERENCE_ARM,
    aggregate,
    complete_trials,
    ranking,
    skill,
    skill_summary,
    stack_horizons,
)
from .config import ExperimentConfig, load_config
from .plotting import (
    GRID,
    INK,
    INK_SECONDARY,
    INTERVAL_MARKER_AREA,
    LINE_WIDTH,
    MARKER_EDGE,
    MUTED,
    SURFACE,
    ArmStyle,
    _curves,
    _figure,
    _interval_panel,
    _save,
    _target_line,
    simulator_styles,
)
from .rebalancing_report import (
    METRIC_LABELS,
    PERCENT,
    comparisons,
    complete_runs,
    per_trial_frame,
    summary,
)
from .regimes import (
    OVERALL,
    REGIMES,
    check_prices,
    origin_features,
    regime_readings,
    terciles,
)
from .report import ReportOutputs, _check_scores_recorded
from .runner import load_prices
from .scores import PORTFOLIO_CRPS, TRAILING_STD_WINDOW
from .table import read_results

# The simulators as the post names them.
BLOG_NAMES = {
    "no_fit": "Historical",
    "gaussian": "Gaussian",
    "kde": "KDE",
    "jointfm": "JointFM",
}
NEXT_DAY = 1
# (title, what the score reads) per ranking score; the variogram score that
# ``analysis.RANKING_SCORES`` also ranks by is left to the report.
BLOG_RANKING_SCORES = {
    "energy_score": ("Energy score", "all assets of the universe jointly"),
    PORTFOLIO_CRPS: ("Portfolio CRPS", "random portfolios of the universe's assets"),
}
BLOG_SKILL_SCORES = {
    "energy_score": ("Energy score", "all assets of the universe jointly"),
    "crps": ("CRPS", "each asset on its own"),
}
REGIME_FEATURE = "volatility_ratio"
REGIME_LABELS = {
    OVERALL: "All origins",
    "low": "Volatility\nfalling",
    "middle": "Volatility\nsteady",
    "high": "Volatility\nrising",
}
# (title, axis label, target) per reading of the regime figure.
REGIME_METRICS = {
    "spread": (
        "Scenario width",
        f"scenario std over the last {TRAILING_STD_WINDOW} days' volatility",
        1.0,
    ),
    "cover_90": (
        "90 % interval coverage",
        "share of realized returns inside the interval",
        0.9,
    ),
}
PERFORMANCE_METRICS = (
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
)
# Axis label of each reading's level; ``METRIC_LABELS`` labels its differences.
LEVEL_UNITS = {
    "annualized_return": "% per year",
    "annualized_volatility": "% per year",
    "sharpe_ratio": "annualized",
    "sortino_ratio": "annualized, downside risk only",
    "max_drawdown": "% below the running peak",
    "calmar_ratio": "annualized return over max drawdown",
}
# Display decimals of a level: percentages to a tenth, ratios to a hundredth.
PERCENT_DECIMALS = 1
RATIO_DECIMALS = 2
# Sized for a figure shown about 700 px wide in a post.
HEADLINE_SIZE = 15
TITLE_SIZE = 12
TEXT_SIZE = 10.5
SUBTITLE_SIZE = 9.5
TITLE_PAD = 22
# Vertical gap between two end labels, as a share of the axis height.
LABEL_GAP = 0.07
# Room right of the longest horizon for the end labels, as a factor on it.
END_LABEL_ROOM = 2.6
# Share of a regime column that its arms spread across.
REGIME_GROUP_WIDTH = 0.6
# Steps between two labeled ticks of the growth figure.
GROWTH_TICK_STEPS = 3
# Room right of the widest interval for its value label, as a share of the range.
VALUE_LABEL_ROOM = 0.3
# The row of risk_return.csv that holds cash at the risk-free rate.
RISK_FREE_ROW = "risk_free"
# Room right of the largest volatility, and above and below the drawn returns,
# as a share of the drawn range.
FRONTIER_ROOM = 0.12
# Opacity of the area below the frontier with cash.
FRONTIER_SHADE = 0.1


@dataclass(frozen=True)
class BlogStyle(ArmStyle):
    """An arm drawn under its simulator's blog name, filled whatever its device."""

    @property
    def filled(self) -> bool:
        return True

    @property
    def inline_label(self) -> str:
        return BLOG_NAMES[self.fit_type]


def blog_styles(per_trial: pd.DataFrame) -> list[BlogStyle]:
    """One style per scenario simulator in simulator order, its GPU arm where it has two.

    Raises
    ------
    ValueError
        If ``REFERENCE_ARM`` is not among the arms, since every blog figure
        reads the simulators against replaying the window.
    """
    # simulator_styles orders CPU before GPU, so a GPU arm replaces its CPU twin.
    by_simulator = {style.fit_type: style for style in simulator_styles(per_trial)}
    styles = [
        BlogStyle(style.name, style.fit_type, style.device)
        for style in by_simulator.values()
    ]
    if REFERENCE_ARM not in {style.name for style in styles}:
        raise ValueError(
            f"The blog reads every simulator against {REFERENCE_ARM}, which the "
            f"results do not contain."
        )
    return styles


def _of_arms(per_trial: pd.DataFrame, styles: Sequence[BlogStyle]) -> pd.DataFrame:
    arms = per_trial.index.get_level_values("arm")
    return per_trial[arms.isin([style.name for style in styles])]


def ranking_table(per_trial: pd.DataFrame, styles: Sequence[BlogStyle]) -> pd.DataFrame:
    """``analysis.ranking`` among the drawn simulators, for ``BLOG_RANKING_SCORES``."""
    drawn = _of_arms(per_trial, styles)
    ranks = ranking(drawn)
    ranks = ranks[ranks["score"].isin(BLOG_RANKING_SCORES)]
    trials = drawn.index.get_level_values("trial").nunique()
    return ranks.assign(trials=trials).reset_index(drop=True)


def skill_table(
    per_trial: dict[int, pd.DataFrame], styles: Sequence[BlogStyle]
) -> pd.DataFrame:
    """Median skill against ``REFERENCE_ARM`` per horizon, score, and drawn simulator."""
    summaries = {
        horizon: skill_summary(skill(_of_arms(table, styles)))
        for horizon, table in per_trial.items()
    }
    stacked = stack_horizons(summaries)
    return stacked[stacked["score"].isin(BLOG_SKILL_SCORES)].reset_index(drop=True)


def regime_table(
    readings: pd.DataFrame, bounds: pd.DataFrame, styles: Sequence[BlogStyle]
) -> pd.DataFrame:
    """The ``REGIME_METRICS`` of ``regime_readings``' summary over all origins and
    per tercile of ``REGIME_FEATURE``, for the drawn simulators, with each
    tercile's feature bounds."""
    rows = readings[
        readings["feature"].isin([OVERALL, REGIME_FEATURE])
        & readings["metric"].isin(REGIME_METRICS)
        & readings["arm"].isin([style.name for style in styles])
    ]
    tercile_bounds = bounds.loc[
        bounds["feature"] == REGIME_FEATURE, ["regime", "low", "high", "origins"]
    ]
    return rows.merge(tercile_bounds, on="regime", how="left").reset_index(drop=True)


def growth_table(
    frame: pd.DataFrame, styles: Sequence[BlogStyle], step_days: int
) -> pd.DataFrame:
    """Per drawn simulator and step: mean cumulative return and mean drawdown.

    Each run's value starts at 1 and compounds its ``step_returns``; its
    drawdown after a step is the share its value lies below the highest value
    of the steps up to it, read at step ends only, so it can be shallower than
    the run's ``max_drawdown`` over daily values. Both are averaged over the
    runs, and ``trading_day`` counts ``step_days`` per step.

    Raises
    ------
    ValueError
        If the runs of one simulator cover different numbers of steps.
    """
    tables = []
    for style in styles:
        paths = list(frame.loc[frame["arm"] == style.name, "step_returns"])
        if len({len(path) for path in paths}) != 1:
            raise ValueError(f"The runs of {style.name} cover different step counts.")
        returns = np.array(paths, dtype=float)
        values = np.cumprod(
            np.concatenate([np.zeros((len(returns), 1)), returns], axis=1) + 1, axis=1
        )
        drawdowns = 1 - values / np.maximum.accumulate(values, axis=1)
        steps = np.arange(values.shape[1])
        tables.append(
            pd.DataFrame(
                {
                    "arm": style.name,
                    "step": steps,
                    "trading_day": steps * step_days,
                    "cumulative_return": values.mean(axis=0) - 1,
                    "drawdown": drawdowns.mean(axis=0),
                    "trials": len(values),
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def performance_table(
    per_trial: pd.DataFrame, styles: Sequence[BlogStyle]
) -> pd.DataFrame:
    """Per drawn simulator and reading of ``PERFORMANCE_METRICS``: level and paired difference.

    ``mean``, ``ci_low``, and ``ci_high`` are ``rebalancing_report.summary``'s
    mean over trials and its interval. Against ``REFERENCE_ARM`` each other
    simulator gets ``rebalancing_report.comparisons``' median paired
    difference (``difference``, ``difference_ci_low``, ``difference_ci_high``,
    ``p_value``) and ``trials_better``, the trials in which it did better in
    the reading's better direction of ``METRIC_LABELS``; a tie is not better.
    The reference arm's difference columns are empty.
    """
    drawn = _of_arms(per_trial, styles)
    levels = summary(drawn)
    differences = comparisons(drawn).rename(
        columns={
            "median_difference": "difference",
            "ci_low": "difference_ci_low",
            "ci_high": "difference_ci_high",
        }
    )
    better = []
    for metric in PERFORMANCE_METRICS:
        wide = drawn[metric].unstack("arm")
        change = wide.drop(columns=REFERENCE_ARM).sub(wide[REFERENCE_ARM], axis=0)
        wins = change > 0 if METRIC_LABELS[metric][2] == "higher" else change < 0
        better.append(
            wins.sum().rename("trials_better").reset_index().assign(metric=metric)
        )
    table = levels[levels["metric"].isin(PERFORMANCE_METRICS)].merge(
        differences[
            [
                "metric",
                "arm",
                "difference",
                "difference_ci_low",
                "difference_ci_high",
                "p_value",
            ]
        ],
        on=["metric", "arm"],
        how="left",
    )
    return table.merge(
        pd.concat(better, ignore_index=True), on=["metric", "arm"], how="left"
    )


def check_rates(rates_path: Path, identity: dict) -> None:
    """Refuse a risk-free rate file other than the one the backtests' cash earned.

    Raises
    ------
    ValueError
        If the SHA-256 of ``rates_path`` differs from the rebalancing results'
        identity.
    """
    recorded = identity["risk_free_rates"]
    if hashlib.sha256(rates_path.read_bytes()).hexdigest() != recorded["sha256"]:
        raise ValueError(
            f"{rates_path} is not the risk-free rate file {recorded['file']} the "
            f"backtests' cash earned (its SHA-256 differs), so the risk-free "
            f"return cannot be recomputed from it."
        )


def risk_free_return(
    runs: pd.DataFrame, dates: pd.DatetimeIndex, rates: pd.Series
) -> float:
    """Cash's annualized return over each run's trading window, averaged over the trials.

    Cash earns ``risk_free.cash_returns`` on every trading day of ``dates`` after
    the window's first, the days a run's daily returns cover, and compounds to
    an annual rate over those days exactly as ``annualized_return`` does, so the
    two read on one scale. Every arm of a trial trades the same window.

    Raises
    ------
    ValueError
        If the arms of one trial traded different windows, or as
        ``risk_free.cash_returns`` raises for a day without a recent rate.
    """
    columns = ["trading_start", "trading_end"]
    if (runs.groupby("trial")[columns].nunique() > 1).any(axis=None):
        raise ValueError("The arms of one trial traded different windows.")
    annual = []
    for start, end in runs.groupby("trial")[columns].first().itertuples(index=False):
        days = dates[(dates > start) & (dates <= end)]
        growth = np.prod(1 + risk_free.cash_returns(rates, days, "LINEAR"))
        annual.append(growth ** (risk_free.TRADING_DAYS_PER_YEAR / len(days)) - 1)
    return float(np.mean(annual))


def risk_return_table(
    performance: pd.DataFrame, styles: Sequence[BlogStyle], risk_free_annual: float
) -> pd.DataFrame:
    """Per drawn simulator: mean annualized volatility and return, and the frontier with cash.

    The volatility and return are ``performance_table``'s means over trials.
    Mixing a portfolio with cash, which earns ``risk_free_annual`` without
    volatility, reaches every point of the line from cash through the
    portfolio's point; ``excess_per_volatility`` is that line's slope, the mean
    return above ``risk_free_annual`` per unit of mean volatility. ``frontier``
    marks the simulator with the steepest line: no mix of cash with another
    simulator's portfolio rises above it. A last row, ``RISK_FREE_ROW``, holds
    cash itself.
    """
    means = performance.set_index(["metric", "arm"])["mean"]
    table = pd.DataFrame(
        {
            "arm": [style.name for style in styles],
            "annualized_volatility": [
                means["annualized_volatility", style.name] for style in styles
            ],
            "annualized_return": [
                means["annualized_return", style.name] for style in styles
            ],
        }
    )
    table["excess_per_volatility"] = (
        table["annualized_return"] - risk_free_annual
    ) / table["annualized_volatility"]
    table["frontier"] = table.index == table["excess_per_volatility"].idxmax()
    cash = pd.DataFrame(
        [
            {
                "arm": RISK_FREE_ROW,
                "annualized_volatility": 0.0,
                "annualized_return": risk_free_annual,
                "frontier": False,
            }
        ]
    )
    table = pd.concat([table, cash], ignore_index=True)
    table["trials"] = int(performance["trials"].iloc[0])
    return table


def _style_text(figure) -> None:
    for ax in figure.axes:
        ax.tick_params(labelsize=TEXT_SIZE)


def _headline(figure, text: str) -> None:
    figure.suptitle(text, x=0.01, ha="left", color=INK, fontsize=HEADLINE_SIZE)


def _panel_title(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=TITLE_SIZE, pad=TITLE_PAD)
    ax.text(
        0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        color=MUTED,
        fontsize=SUBTITLE_SIZE,
        va="bottom",
    )


def _axis_labels(ax, x: str, y: str) -> None:
    ax.set_xlabel(x, color=INK_SECONDARY, fontsize=TEXT_SIZE)
    ax.set_ylabel(y, color=INK_SECONDARY, fontsize=TEXT_SIZE)


def _legend(
    figure, styles: Sequence[BlogStyle], *, lines: bool, reference: str
) -> None:
    """One entry per simulator below the panels, as a line or as a marker.

    A last entry names the dark horizontal line every panel draws
    (``_target_line``) as ``reference``; a label inside the panels would
    collide with the marks near it.
    """
    handles = [
        Line2D(
            [],
            [],
            color=style.color,
            linewidth=LINE_WIDTH if lines else 0,
            marker="o",
            markersize=8,
            markerfacecolor=style.color,
            markeredgecolor=SURFACE,
            markeredgewidth=MARKER_EDGE,
            label=style.inline_label,
        )
        for style in styles
    ]
    handles.append(Line2D([], [], color=INK_SECONDARY, linewidth=1, label=reference))
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncols=len(handles),
        frameon=False,
        fontsize=TEXT_SIZE,
        labelcolor=INK_SECONDARY,
    )


def _end_labels(ax, labels: Sequence[tuple[float, float, str]]) -> None:
    """Label line ends at (x, y, text), pushed apart so no two labels overlap.

    Call it once the axis limits are final: the gap between labels is
    ``LABEL_GAP`` of the axis height, and the labels keep their order by y.
    """
    low, high = ax.get_ylim()
    gap = LABEL_GAP * (high - low)
    placed: list[float] = []
    for x, y, text in sorted(labels, key=itemgetter(1)):
        placed.append(max(y, placed[-1] + gap) if placed else y)
        ax.annotate(
            text,
            xy=(x, placed[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK_SECONDARY,
            fontsize=TEXT_SIZE,
        )


def plot_ranking(
    ranks: pd.DataFrame, styles: Sequence[BlogStyle], directory: Path
) -> list[Path]:
    """Each simulator's share of trials won per score, best mean rank on top."""
    by_name = {style.name: style for style in styles}
    figure, axes = _figure(12, 0.6 * len(styles) + 2.4, cols=len(BLOG_RANKING_SCORES))
    for ax, (score, (title, reads)) in zip(axes.flat, BLOG_RANKING_SCORES.items()):
        rows = ranks[ranks["score"] == score].set_index("arm")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=1)
        for position, arm in enumerate(rows.index):
            share = rows.loc[arm, "share_won"]
            ax.barh(position, share, height=0.55, color=by_name[arm].color)
            ax.text(
                share + 0.02,
                position,
                f"{PERCENT * share:.0f} % · mean rank {rows.loc[arm, 'mean_rank']:.2f}",
                va="center",
                color=INK_SECONDARY,
                fontsize=TEXT_SIZE,
            )
        ax.set_yticks(
            range(len(rows)), [by_name[arm].inline_label for arm in rows.index]
        )
        ax.invert_yaxis()
        # Room right of a full bar for its label; ticks stop at 100 %.
        ax.set_xlim(0, 1.55)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        _panel_title(
            ax,
            f"{title}: {reads}",
            f"share of {int(rows['trials'].iloc[0])} trials with the best "
            f"next-day score",
        )
    _headline(figure, "Most realistic next-day scenarios per trial")
    _style_text(figure)
    return _save(figure, directory, "ranking")


def plot_skill_horizons(
    skills: pd.DataFrame, styles: Sequence[BlogStyle], directory: Path
) -> list[Path]:
    """Median skill against replaying the window over the horizons, one panel per score."""
    contenders = [style for style in styles if style.name != REFERENCE_ARM]
    reference = next(style for style in styles if style.name == REFERENCE_ARM)
    horizons = sorted(skills["horizon"].unique())
    figure, axes = _figure(12, 5.2, cols=len(BLOG_SKILL_SCORES))
    for ax, (score, (title, reads)) in zip(axes.flat, BLOG_SKILL_SCORES.items()):
        table = skills[skills["score"] == score].rename(
            columns={"median_skill": "estimate"}
        )
        _curves(ax, table, "horizon", contenders)
        _target_line(ax, 0)
        ax.set_xlim(right=horizons[-1] * END_LABEL_ROOM)
        last = table[table["horizon"] == horizons[-1]].set_index("arm")
        _end_labels(
            ax,
            [
                (
                    horizons[-1] * 1.1,
                    last.loc[style.name, "estimate"],
                    style.inline_label,
                )
                for style in contenders
            ],
        )
        _axis_labels(
            ax,
            "forecast horizon (trading days)",
            f"% better than {reference.inline_label}",
        )
        _panel_title(
            ax,
            f"{title}: {reads}",
            "median over trials with its 95 % bootstrap interval",
        )
    _headline(figure, "Score improvement over replaying the window, by horizon")
    _legend(
        figure,
        contenders,
        lines=True,
        reference=f"{reference.inline_label}, replaying the window",
    )
    _style_text(figure)
    return _save(figure, directory, "skill_horizons")


def _regime_ticks(table: pd.DataFrame) -> list[str]:
    ticks = [REGIME_LABELS[OVERALL]]
    for regime in REGIMES:
        row = table[table["regime"] == regime].iloc[0]
        ticks.append(f"{REGIME_LABELS[regime]}\n{row['low']:.2f}–{row['high']:.2f}")
    return ticks


def plot_volatility_regimes(
    table: pd.DataFrame, styles: Sequence[BlogStyle], directory: Path
) -> list[Path]:
    """Spread and coverage over all origins and per volatility tercile, one point per simulator."""
    groups = [OVERALL, *REGIMES]
    offsets = np.linspace(-REGIME_GROUP_WIDTH / 2, REGIME_GROUP_WIDTH / 2, len(styles))
    figure, axes = _figure(12, 5.4, cols=len(REGIME_METRICS))
    for ax, (metric, (title, unit, target)) in zip(axes.flat, REGIME_METRICS.items()):
        ax.grid(axis="y", color=GRID, linewidth=1)
        for offset, style in zip(offsets, styles):
            rows = table[(table["metric"] == metric) & (table["arm"] == style.name)]
            rows = rows.set_index("regime").loc[groups]
            positions = np.arange(len(groups)) + offset
            ax.vlines(
                positions,
                rows["ci_low"],
                rows["ci_high"],
                color=style.color,
                linewidth=LINE_WIDTH,
            )
            ax.scatter(
                positions,
                rows["mean"],
                s=INTERVAL_MARKER_AREA,
                facecolor=style.color,
                edgecolor=SURFACE,
                linewidth=MARKER_EDGE,
                zorder=3,
            )
        _target_line(ax, target)
        # Separates the overall column from the terciles.
        ax.axvline(0.5, color=GRID, linewidth=1)
        ax.set_xticks(range(len(groups)), _regime_ticks(table))
        if metric.startswith("cover_"):
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        _axis_labels(ax, "", unit)
        _panel_title(
            ax,
            title,
            f"next day; thirds of origins by the last {TRAILING_STD_WINDOW} days' "
            f"volatility over the window's",
        )
    _headline(figure, "Scenario width and coverage as volatility changes")
    _legend(figure, styles, lines=False, reference="target")
    _style_text(figure)
    return _save(figure, directory, "volatility_regimes")


def plot_growth(
    growth: pd.DataFrame, styles: Sequence[BlogStyle], directory: Path
) -> list[Path]:
    """Mean cumulative return and mean drawdown after every step, one line per simulator."""
    figure, axes = _figure(10, 7.5, rows=2)
    top, bottom = axes[:, 0]
    bottom.sharex(top)
    labels = []
    for style in styles:
        rows = growth[growth["arm"] == style.name]
        days = rows["trading_day"].to_numpy()
        returns = PERCENT * rows["cumulative_return"].to_numpy()
        for ax, values in (
            (top, returns),
            (bottom, -PERCENT * rows["drawdown"].to_numpy()),
        ):
            ax.plot(
                days,
                values,
                color=style.color,
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
                zorder=3,
            )
        labels.append(
            (days[-1], returns[-1], f"{style.inline_label} {returns[-1]:+.1f} %")
        )
    _target_line(top, 0)
    _target_line(bottom, 0)
    steps = growth["trading_day"].unique()
    top.set_xticks(steps[::GROWTH_TICK_STEPS])
    top.set_xlim(steps[0], steps[-1] * 1.25)
    top.tick_params(labelbottom=False)
    _end_labels(top, labels)
    trials = int(growth["trials"].iloc[0])
    _axis_labels(top, "", "cumulative return (%)")
    _panel_title(
        top, "Cumulative return", f"mean over {trials} trials at every rebalancing date"
    )
    _axis_labels(bottom, "trading days since the backtest started", "drawdown (%)")
    _panel_title(
        bottom,
        "Drawdown from the running peak",
        f"mean over {trials} trials at every rebalancing date",
    )
    _headline(figure, "Rebalanced portfolio over the backtest")
    _legend(figure, styles, lines=True, reference="starting value")
    _style_text(figure)
    return _save(figure, directory, "growth")


def _level(performance: pd.DataFrame, metric: str, arm: str) -> np.ndarray:
    """(mean, interval low, interval high) of a reading, in display units."""
    row = performance[(performance["metric"] == metric) & (performance["arm"] == arm)]
    scale = METRIC_LABELS[metric][3]
    return scale * row[["mean", "ci_low", "ci_high"]].to_numpy(dtype=float)[0]


def plot_risk_return(
    table: pd.DataFrame, styles: Sequence[BlogStyle], directory: Path
) -> list[Path]:
    """Each simulator and cash as a dot at mean annualized volatility and return, with the frontier.

    The frontier is the line from cash through the simulator ``frontier`` marks
    in ``risk_return_table``: solid where a mix of its portfolio with cash
    reaches it, dashed beyond, where the mix borrows at the risk-free rate. The
    area below the line is shaded.
    """
    figure, axes = _figure(8, 6.5)
    ax = axes[0, 0]
    ax.grid(axis="x", color=GRID, linewidth=1)
    points = (
        PERCENT * table.set_index("arm")[["annualized_volatility", "annualized_return"]]
    )
    cash = points.loc[RISK_FREE_ROW, "annualized_return"]
    best = next(
        style
        for style in styles
        if table.loc[table["arm"] == style.name, "frontier"].item()
    )
    best_volatility, best_return = points.loc[best.name]
    slope = (best_return - cash) / best_volatility
    right = (1 + FRONTIER_ROOM) * points["annualized_volatility"].max()
    edge = cash + slope * right
    drawn = [*points["annualized_return"], edge]
    room = FRONTIER_ROOM * (max(drawn) - min(drawn))
    bottom, top = min(drawn) - room, max(drawn) + room
    ax.fill_between(
        [0, right],
        [cash, edge],
        bottom,
        color=best.color,
        alpha=FRONTIER_SHADE,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        [0, best_volatility],
        [cash, best_return],
        color=best.color,
        linewidth=LINE_WIDTH,
        zorder=2,
    )
    ax.plot(
        [best_volatility, right],
        [best_return, edge],
        color=best.color,
        linewidth=LINE_WIDTH,
        linestyle="--",
        zorder=2,
    )
    ax.annotate(
        f"cash mixed with {best.inline_label}",
        xy=(right, edge),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=best.color,
        fontsize=TEXT_SIZE,
    )
    dots = [(style.name, style.color, style.inline_label) for style in styles] + [
        (RISK_FREE_ROW, INK_SECONDARY, f"cash, risk-free {cash:.1f} %")
    ]
    for arm, color, label in dots:
        volatility, annual = points.loc[arm]
        ax.scatter(
            [volatility],
            [annual],
            s=INTERVAL_MARKER_AREA,
            facecolor=color,
            edgecolor=SURFACE,
            linewidth=MARKER_EDGE,
            zorder=3,
        )
        ax.annotate(
            label,
            xy=(volatility, annual),
            xytext=(8, 6),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=TEXT_SIZE,
        )
    ax.set_xlim(-FRONTIER_ROOM * right / 4, right)
    ax.set_ylim(bottom, top)
    trials = int(table["trials"].iloc[0])
    _axis_labels(ax, "annualized volatility (%)", "annualized return (%)")
    _panel_title(
        ax,
        "Return against risk, with cash",
        f"mean over {trials} trials; up and left is better",
    )
    figure.supxlabel(
        "The line mixes cash at the risk-free rate with the portfolio whose line "
        "from cash is steepest,\ndashed where the mix borrows cash. A dot in the "
        "shaded area earns less than the mix of equal volatility.",
        color=MUTED,
        fontsize=SUBTITLE_SIZE,
    )
    _headline(figure, "Rebalanced portfolios: return against volatility, with cash")
    _style_text(figure)
    return _save(figure, directory, "risk_return")


def plot_performance(
    performance: pd.DataFrame, styles: Sequence[BlogStyle], directory: Path
) -> list[Path]:
    """One panel per reading of ``PERFORMANCE_METRICS``: each simulator's mean and interval.

    A vertical line marks ``REFERENCE_ARM``'s mean, and each other simulator's
    label counts the trials in which it did better than the reference.
    """
    cols = 3
    rows = -(-len(PERFORMANCE_METRICS) // cols)
    figure, axes = _figure(15, rows * (0.75 * len(styles) + 2.2), rows=rows, cols=cols)
    reference = next(style for style in styles if style.name == REFERENCE_ARM)
    for ax, metric in zip(axes.flat, PERFORMANCE_METRICS):
        title, _, better, scale = METRIC_LABELS[metric]
        levels = [_level(performance, metric, style.name) for style in styles]
        _interval_panel(
            ax,
            [(style, *level) for style, level in zip(styles, levels)],
            reference=_level(performance, metric, REFERENCE_ARM)[0],
            reference_label=reference.inline_label,
        )
        counts = performance[performance["metric"] == metric].set_index("arm")
        ax.set_yticks(
            range(len(styles)),
            [
                style.inline_label
                if style.name == REFERENCE_ARM
                else f"{style.inline_label}\nbetter in "
                f"{int(counts.loc[style.name, 'trials_better'])} of "
                f"{int(counts.loc[style.name, 'trials'])}"
                for style in styles
            ],
        )
        unit, decimals = (
            (" %", PERCENT_DECIMALS) if scale == PERCENT else ("", RATIO_DECIMALS)
        )
        low = min(level[1] for level in levels)
        high = max(level[2] for level in levels)
        ax.set_xlim(low - 0.05 * (high - low), high + VALUE_LABEL_ROOM * (high - low))
        for position, level in enumerate(levels):
            ax.annotate(
                f"{level[0]:.{decimals}f}{unit}",
                xy=(level[2], position),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=INK_SECONDARY,
                fontsize=TEXT_SIZE,
            )
        ax.set_xlabel(LEVEL_UNITS[metric], color=INK_SECONDARY, fontsize=TEXT_SIZE)
        _panel_title(ax, title, f"{better} is better")
    for ax in list(axes.flat)[len(PERFORMANCE_METRICS) :]:
        ax.set_visible(False)
    trials = int(performance["trials"].iloc[0])
    figure.supxlabel(
        f"Mean over {trials} trials with its 95 % bootstrap interval; each label "
        f"counts the trials in which the simulator did better than "
        f"{reference.inline_label}.",
        color=MUTED,
        fontsize=SUBTITLE_SIZE,
    )
    _headline(figure, "Performance and risk of the rebalanced portfolios")
    _style_text(figure)
    return _save(figure, directory, "performance")


def write_blog(config: ExperimentConfig) -> ReportOutputs:
    """Compute the blog's readings from a configuration's results and write its figures.

    The scenario figures read ``config.output`` and the price file it was
    measured on, over the trials in which every arm has every origin at every
    horizon; the rebalancing figures read ``config.rebalancing.output``, over
    the trials in which every arm finished, and the risk-free rate file the
    backtests' cash earned. Everything goes to ``config.blog_dir``.

    Raises
    ------
    ValueError
        If the scenario results hold no scenario or portfolio scores, either
        results file was measured on another price file, the rebalancing
        results earned cash at another risk-free rate file, the scenario
        results lack ``REFERENCE_ARM``, have too few complete trials or a
        volatility tercile with too few trials, or the rebalancing results
        have too few complete trials.
    """
    frame, metadata = read_results(config.output)
    identity = metadata.identity
    _check_scores_recorded(identity)
    check_prices(config.prices, identity)
    experiment = identity["experiment"]
    frame, excluded = complete_trials(
        frame,
        tuple(identity["arms"]),
        experiment["origins"],
        len(experiment["horizons"]),
    )
    per_trial = {
        horizon: aggregate(frame[frame["horizon"] == horizon])
        for horizon in experiment["horizons"]
    }
    styles = blog_styles(per_trial[NEXT_DAY])
    prices = load_prices(config.prices)
    features = origin_features(prices, frame, experiment)
    regimes, bounds = terciles(features)
    readings = regime_readings(
        frame[frame["horizon"] == NEXT_DAY], regimes, ("summary",)
    )

    runs, run_metadata = read_results(config.rebalancing.output)
    check_prices(config.prices, run_metadata.identity)
    check_rates(config.rebalancing.risk_free_rates, run_metadata.identity)
    runs = complete_runs(runs, tuple(run_metadata.identity["arms"]))
    run_styles = blog_styles(per_trial_frame(runs))
    step_days = run_metadata.identity["rebalancing"]["look_forward_window"]
    performance = performance_table(per_trial_frame(runs), run_styles)
    cash = risk_free_return(
        runs,
        prices.index,
        risk_free.load_risk_free_rates(config.rebalancing.risk_free_rates),
    )

    tables = {
        "ranking": ranking_table(per_trial[NEXT_DAY], styles),
        "skill_horizons": skill_table(per_trial, styles),
        "volatility_regimes": regime_table(readings["summary"], bounds, styles),
        "growth": growth_table(runs, run_styles, step_days),
        "risk_return": risk_return_table(performance, run_styles, cash),
        "performance": performance,
    }
    directory = config.blog_dir
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for name, table in tables.items():
        path = directory / f"{name}.csv"
        table.to_csv(path, index=False)
        files.append(path)
    files += plot_ranking(tables["ranking"], styles, directory)
    files += plot_skill_horizons(tables["skill_horizons"], styles, directory)
    files += plot_volatility_regimes(tables["volatility_regimes"], styles, directory)
    files += plot_growth(tables["growth"], run_styles, directory)
    files += plot_risk_return(tables["risk_return"], run_styles, directory)
    files += plot_performance(tables["performance"], run_styles, directory)
    return ReportOutputs(files=files, excluded_trials=excluded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.simulator_comparison.blog",
        description="Blog figures of a configuration's scenario and rebalancing "
        "results, written to <output stem>_blog beside its results; runs nothing.",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="YAML configuration file."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    config = load_config(build_parser().parse_args(argv).config)
    outputs = write_blog(config)
    if outputs.excluded_trials:
        print(
            f"Left out {len(outputs.excluded_trials)} incomplete scenario trials: "
            f"{outputs.excluded_trials}"
        )
    print(f"Blog figures finished: {len(outputs.files)} files in {config.blog_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

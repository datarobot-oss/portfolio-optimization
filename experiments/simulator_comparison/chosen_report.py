# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Risk of the chosen portfolios drawn over the size curves of random portfolios.

``rebalancing_chosen_risk.csv`` (``rebalancing_report.chosen_risk``) gives,
per arm, how often the portfolios the optimizer chose breached their scenario
VaR and how their realized volatility compares with the scenario volatility.
An optimizer seeks out the portfolios whose risk its scenarios understate
most, so the question is how these readings compare with those of random
portfolios of the same size under the same arm's scenarios. This module draws
both readings over the next-day curves of the random long-only and long-short
portfolios of the scenario comparison (``portfolio_analysis.size_curves``),
recomputed over the trials the backtest ran. Each arm's chosen portfolios sit
at their mean effective number of holdings, which is ``n`` for ``n`` equal
positions and so reads on the curves' portfolio-size axis. The spread is drawn
on the curves' scale, scenario over realized, the reciprocal of the recorded
volatility ratio.

The curves come from the scenario results file that
``rebalancing.scenario_results`` names. A twin config that runs only the
backtest names the file of the config it is twin to, because the random
portfolios depend on the trials' universes and the weight, cash, and leverage
bounds, which the twins share, and not on the risk aversion.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .analysis import aggregate, complete_trials
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
    _figure,
    _save,
    _title,
)
from .portfolio_analysis import breach_reading, size_curves
from .portfolio_report import KIND_LABELS, READING_LABELS, _reading_panel
from .report import _check_scores_recorded
from .scores import PORTFOLIO_CRPS, PORTFOLIO_VAR_LEVELS
from .table import read_results

# The portfolio kinds whose weights are drawn within the optimizer's bounds,
# as the chosen portfolios' are.
RANDOM_KINDS = ("long_only", "long_short")
SPREAD_READING = "spread_ratio"
# The bounds of the random portfolios that the scenario results record.
BOUND_KEYS = ("w_min", "w_max", "c_min", "c_max", "L_tar")
POINT_COLUMNS = [
    "arm",
    "reading",
    "holdings",
    "estimate",
    "ci_low",
    "ci_high",
    "band_low",
    "band_high",
]
# A chosen point's band spans this factor either side of it on the log axis,
# narrower than a size's band so the two stay apart where they meet.
CHOSEN_BAND_HALF_WIDTH = 1.1
CHOSEN_HATCH = "////"
CHOSEN_MARKER = "D"
# Room beyond the outermost size or chosen point, as ``plotting._log_axis`` leaves.
AXIS_MARGIN = 1.4
# The explanations of the marks and bands are too long for one legend row.
MARK_LEGEND_COLUMNS = 2
BREACH_SUBTITLE = "share of days; bands: range of a correct VaR"
SPREAD_SUBTITLE = "scenario over realized; below 1 the scenarios are too narrow"


def backtest_universes(frame: pd.DataFrame) -> pd.Series:
    """Each backtested trial's ``assets_hash``, indexed by trial."""
    return frame.groupby("trial")["assets_hash"].first()


def _portfolio_bounds(identity: dict) -> dict:
    (bounds,) = [
        measurement["parameters"]["bounds"]
        for measurement in identity["measurements"]
        if PORTFOLIO_CRPS in measurement["columns"]
    ]
    return bounds


def next_day_trials(
    scenario_results: Path, identity: dict, universes: pd.Series
) -> pd.DataFrame:
    """The next-day per-trial scenario readings of the backtested trials.

    Parameters
    ----------
    scenario_results : Path
        Scenario-comparison results file, ``rebalancing.scenario_results``.
    identity : dict
        Identity of the rebalancing results file.
    universes : pd.Series
        ``assets_hash`` of each backtested trial, indexed by trial
        (``backtest_universes``).

    Returns
    -------
    pd.DataFrame
        ``aggregate`` of the next-day rows of exactly the trials of
        ``universes``, one row per (trial, arm) of the scenario comparison.

    Raises
    ------
    ValueError
        If the file does not exist, holds no portfolio scores, was measured
        on other prices, drew its random portfolios within other bounds than
        the optimizer's, has fewer complete trials than the report needs,
        lacks a backtested trial or holds it incomplete, or trades another
        universe in one.
    """
    if not scenario_results.is_file():
        raise ValueError(
            f"The chosen-risk figure reads the size curves of {scenario_results}, "
            f"which does not exist. Run the scenario comparison that writes it "
            f"(task experiments:run with the config whose output it is), then "
            f"task experiments:rebalance-report."
        )
    frame, metadata = read_results(scenario_results)
    scenario = metadata.identity
    _check_scores_recorded(scenario)
    if scenario["prices"]["sha256"] != identity["prices"]["sha256"]:
        raise ValueError(
            f"{scenario_results} was measured on another price file than the "
            f"backtest, so its trials are not the backtest's."
        )
    bounds = _portfolio_bounds(scenario)
    optimizer = {key: identity["rebalancing"]["cvar"][key] for key in BOUND_KEYS}
    if bounds != optimizer:
        raise ValueError(
            f"{scenario_results} drew its random portfolios within {bounds}, the "
            f"optimizer chose within {optimizer}; the two do not compare."
        )
    experiment = scenario["experiment"]
    complete, _ = complete_trials(
        frame,
        tuple(scenario["arms"]),
        experiment["origins"],
        len(experiment["horizons"]),
    )
    hashes = complete.groupby("trial")["assets_hash"].first()
    missing = sorted(int(trial) for trial in universes.index.difference(hashes.index))
    if missing:
        raise ValueError(
            f"{scenario_results} holds no complete record of the backtested "
            f"trials {missing}; finish its scenario comparison first."
        )
    other = [
        int(trial) for trial, assets in universes.items() if hashes[trial] != assets
    ]
    if other:
        raise ValueError(
            f"{scenario_results} trades other universes than the backtest in "
            f"trials {other}."
        )
    rows = complete[
        complete["trial"].isin(universes.index) & (complete["horizon"] == 1)
    ]
    return aggregate(rows)


def random_size_curves(per_trial: pd.DataFrame) -> pd.DataFrame:
    """``size_curves`` of the random long-only and long-short portfolios.

    Raises
    ------
    ValueError
        If the bounds admit neither kind at any recorded size.
    """
    curves = size_curves(per_trial)
    curves = curves[curves["kind"].isin(RANDOM_KINDS)].reset_index(drop=True)
    if curves.empty:
        raise ValueError(
            "The scenario results hold no random long-only or long-short "
            "portfolio to draw the chosen portfolios over."
        )
    return curves


def chosen_points(chosen: pd.DataFrame) -> pd.DataFrame:
    """Each arm's two chosen-portfolio readings on the size curves' scale.

    Parameters
    ----------
    chosen : pd.DataFrame
        ``rebalancing_report.chosen_risk``, one row per arm.

    Returns
    -------
    pd.DataFrame
        Two rows per arm, at ``holdings``, its mean effective number of
        holdings: the breach rate, named as the size curves' breach reading
        at the VaR level, with its bootstrap interval and the binomial band
        of a correct VaR; and ``spread_ratio``, the reciprocal of the
        volatility ratio, whose interval is the reciprocal interval, since
        the reciprocal is monotone.

    Raises
    ------
    ValueError
        If the arms were read at more than one VaR level, or at one for
        which the size curves have no breach reading.
    """
    levels = chosen["var_level"].unique()
    if len(levels) != 1:
        raise ValueError(f"The chosen portfolios hold VaR levels {list(levels)}.")
    (level,) = levels
    if not np.isclose(level, PORTFOLIO_VAR_LEVELS).any():
        raise ValueError(
            f"The size curves read breaches at VaR levels "
            f"{list(PORTFOLIO_VAR_LEVELS)}, the chosen portfolios at {level}."
        )
    breaches = pd.DataFrame(
        {
            "arm": chosen["arm"],
            "reading": breach_reading(level),
            "holdings": chosen["effective_holdings"],
            "estimate": chosen["breach_rate"],
            "ci_low": chosen["breach_ci_low"],
            "ci_high": chosen["breach_ci_high"],
            "band_low": chosen["breach_band_low"],
            "band_high": chosen["breach_band_high"],
        }
    )
    spreads = pd.DataFrame(
        {
            "arm": chosen["arm"],
            "reading": SPREAD_READING,
            "holdings": chosen["effective_holdings"],
            "estimate": 1 / chosen["volatility_ratio"],
            "ci_low": 1 / chosen["volatility_ratio_ci_high"],
            "ci_high": 1 / chosen["volatility_ratio_ci_low"],
            "band_low": np.nan,
            "band_high": np.nan,
        }
    )
    return pd.concat([breaches, spreads], ignore_index=True)[POINT_COLUMNS]


def point_gid(arm: str) -> str:
    """Artist id of an arm's chosen-portfolio marker, also its group id in the SVG."""
    return f"chosen-point-{arm}"


def band_gid(arm: str) -> str:
    """Artist id of the band of a correct VaR at an arm's chosen portfolios."""
    return f"chosen-band-{arm}"


def _chosen(ax, points: pd.DataFrame, styles: Sequence[ArmStyle]) -> None:
    """Each arm's chosen point at its holdings, with its interval and band."""
    for style in styles:
        point = points.loc[style.name]
        x = point["holdings"]
        if not np.isnan(point["band_low"]):
            ax.fill_between(
                [x / CHOSEN_BAND_HALF_WIDTH, x * CHOSEN_BAND_HALF_WIDTH],
                point["band_low"],
                point["band_high"],
                facecolor="none",
                edgecolor=MUTED,
                hatch=CHOSEN_HATCH,
                linewidth=0,
                zorder=1,
                gid=band_gid(style.name),
            )
        ax.vlines(
            [x],
            point["ci_low"],
            point["ci_high"],
            color=style.color,
            linewidth=LINE_WIDTH,
            zorder=5,
        )
        ax.scatter(
            [x],
            [point["estimate"]],
            marker=CHOSEN_MARKER,
            s=INTERVAL_MARKER_AREA,
            facecolor=style.color if style.filled else SURFACE,
            edgecolor=style.color,
            linewidth=MARKER_EDGE,
            zorder=6,
            gid=point_gid(style.name),
        )


def _legends(figure, styles: Sequence[ArmStyle]) -> None:
    """The arms in a column right of the panels, the marks and bands below."""
    arms = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=style.color if style.filled else SURFACE,
            markeredgecolor=style.color,
            markeredgewidth=MARKER_EDGE,
            label=style.inline_label,
        )
        for style in styles
    ]
    marks = [
        Line2D(
            [],
            [],
            color=MUTED,
            marker="o",
            linewidth=LINE_WIDTH,
            label="random portfolios of each size",
        ),
        Line2D(
            [],
            [],
            color=MUTED,
            marker=CHOSEN_MARKER,
            linestyle="",
            markersize=8,
            label="chosen portfolios, at their mean effective holdings",
        ),
        Patch(
            facecolor=GRID,
            label="range of a correct VaR over the random portfolios' origins",
        ),
        Patch(
            facecolor="none",
            edgecolor=MUTED,
            hatch=CHOSEN_HATCH,
            linewidth=0,
            label="range of a correct VaR over the chosen portfolios' held days",
        ),
    ]
    for handles, loc, ncols in (
        (arms, "outside right upper", 1),
        (marks, "outside lower center", MARK_LEGEND_COLUMNS),
    ):
        figure.legend(
            handles=handles,
            loc=loc,
            ncols=ncols,
            frameon=False,
            fontsize=8,
            labelcolor=INK_SECONDARY,
        )


def chosen_risk_figure(
    curves: pd.DataFrame,
    points: pd.DataFrame,
    curve_styles: Sequence[ArmStyle],
    styles: Sequence[ArmStyle],
) -> Figure:
    """The chosen points of ``styles`` over the curves of ``curve_styles``.

    Rows are the breach rate and the spread ratio, columns the random kinds
    that ``curves`` holds. Each panel draws the curves as
    ``portfolio_report.plot_portfolio_sizes`` does, and each arm's chosen
    point as a diamond at its holdings with its interval; on the breach
    panels a hatched band at the point marks the range of a correct VaR.
    The chosen portfolios are not of either kind; both kinds' curves are
    there to read them against.
    """
    points = points.set_index("arm")
    readings = list(dict.fromkeys(points["reading"]))
    kinds = [kind for kind in RANDOM_KINDS if kind in set(curves["kind"])]
    figure, axes = _figure(
        6.0 * len(kinds), 3.8 * len(readings) + 0.8, rows=len(readings), cols=len(kinds)
    )
    for row, reading in enumerate(readings):
        drawn = points[points["reading"] == reading]
        subtitle = SPREAD_SUBTITLE if reading == SPREAD_READING else BREACH_SUBTITLE
        for column, kind in enumerate(kinds):
            ax = axes[row, column]
            cell = curves[(curves["kind"] == kind) & (curves["reading"] == reading)]
            _reading_panel(ax, cell, "size", reading, curve_styles)
            _chosen(ax, drawn, styles)
            sizes = cell["size"]
            ax.set_xlim(
                min(sizes.min(), drawn["holdings"].min()) / AXIS_MARGIN,
                max(sizes.max(), drawn["holdings"].max()) * AXIS_MARGIN,
            )
            ax.set_xlabel(
                "assets per portfolio; effective holdings for the chosen ones",
                fontsize=8,
            )
            _title(ax, f"{READING_LABELS[reading][0]} · {KIND_LABELS[kind]}", subtitle)
    figure.suptitle(
        "Risk of the chosen portfolios over the next-day size curves of random "
        "portfolios\nboth over the backtested trials: point and 95 % bootstrap "
        "interval over trials",
        color=INK,
        fontsize=10,
    )
    legend = {style.name: style for style in [*styles, *curve_styles]}
    _legends(figure, list(legend.values()))
    return figure


def curve_styles(
    scenario_styles: Sequence[ArmStyle], styles: Sequence[ArmStyle]
) -> list[ArmStyle]:
    """The scenario arms whose simulator one of the drawn arms ``styles`` runs.

    A figure that leaves an arm out leaves its curve out too, and an arm's
    curve stays when the two comparisons ran its simulator on other devices.
    """
    simulators = {style.fit_type for style in styles}
    return [style for style in scenario_styles if style.fit_type in simulators]


def plot_chosen_risk(
    chosen: pd.DataFrame,
    curves: pd.DataFrame,
    scenario_styles: Sequence[ArmStyle],
    styles: Sequence[ArmStyle],
    directory: Path,
) -> list[Path]:
    """Draw ``chosen_risk`` of the arms of ``styles`` over ``curves``.

    ``scenario_styles`` are the arms of the scenario comparison, of which
    ``curve_styles`` picks the curves to draw.
    """
    figure = chosen_risk_figure(
        curves,
        chosen_points(chosen[chosen["arm"].isin([style.name for style in styles])]),
        curve_styles(scenario_styles, styles),
        styles,
    )
    return _save(figure, directory, "rebalancing_chosen_risk")

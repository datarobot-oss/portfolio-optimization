# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Figures of the portfolio readings as curves over the portfolio size and horizon.

``write_report`` calls them with the tables of ``portfolio_analysis``, and
with the effects on the portfolio readings when the results hold the
dependence swap arms or the independence control; they follow the shared look
of ``plotting``.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from .analysis import REFERENCE_ARM
from .plotting import (
    GRID,
    INK,
    SURFACE,
    ArmStyle,
    EffectStyle,
    _arm_legend,
    _curves,
    _effect_legend,
    _figure,
    _save,
    _target_line,
    _title,
)
from .portfolios import KINDS
from .scores import PIT_BINS

KIND_LABELS = {
    "equal": "Equal weights, fully invested",
    "long_only": "Random long-only within the bounds",
    "long_short": "Random long-short within the bounds",
}
# (title, unit) per reading of ``size_curves``.
READING_LABELS = {
    "breach_05": (
        "95 % VaR breach rate",
        "share of portfolio-days; gray: range of a calibrated forecast",
    ),
    "breach_01": (
        "99 % VaR breach rate",
        "share of portfolio-days; gray: range of a calibrated forecast",
    ),
    "spread_ratio": (
        "Scenario over realized spread",
        "1 / sqrt(mean squared standardized error); below 1 is too narrow",
    ),
    "crps_skill": (
        "Portfolio CRPS skill",
        f"% better than {REFERENCE_ARM}, median over trials",
    ),
}
PERCENT_READINGS = ("breach_05", "breach_01")
# (title, unit) per portfolio reading of ``swap_effects`` and
# ``independence_effects``, which difference the arms' own per-trial readings
# rather than those ``size_curves`` derives.
EFFECT_READING_LABELS = {
    "breach_05": ("95 % VaR breach rate", "share of portfolio-days"),
    "breach_01": ("99 % VaR breach rate", "share of portfolio-days"),
    "crps": (
        "Portfolio CRPS",
        "portfolio trailing volatility × √h units; lower is better",
    ),
    "z2": (
        "Squared standardized error",
        "mean z²; above 0 means narrower scenarios",
    ),
}
# A position's calibrated band spans this factor either side of it on the log axis.
BAND_HALF_WIDTH = 1.2
# Inches that keep the suptitle of a horizon figure with few sizes unclipped.
MIN_HORIZON_FIGURE_WIDTH = 12.0


def _kinds(curves: pd.DataFrame) -> list[str]:
    return [kind for kind in KINDS if kind in set(curves["kind"])]


def _reading_panel(
    ax, cell: pd.DataFrame, x: str, reading: str, styles: Sequence[ArmStyle]
) -> None:
    """One reading of ``size_curves`` over the column ``x``, each arm a curve.

    A breach-rate panel shades the binomial band of a calibrated forecast at
    each position and formats its axis in percent; every panel marks the
    reading's target.
    """
    if reading in PERCENT_READINGS:
        band = cell.groupby(x)[["band_low", "band_high"]].first()
        for position, low, high in band.itertuples():
            ax.fill_between(
                [position / BAND_HALF_WIDTH, position * BAND_HALF_WIDTH],
                low,
                high,
                color=GRID,
                zorder=1,
            )
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    _target_line(ax, cell["target"].iloc[0])
    _curves(ax, cell, x, styles)


def _size_grid(
    curves: pd.DataFrame,
    labels: dict[str, tuple[str, str]],
    draw: Callable[[Axes, pd.DataFrame, str], None],
) -> Figure:
    """One panel per reading of ``labels`` and portfolio kind, over the size.

    Rows are the readings, in ``labels`` order, columns the kinds in
    ``curves``; ``draw`` draws the rows of ``curves`` of one kind and reading
    on its panel.
    """
    kinds = _kinds(curves)
    figure, axes = _figure(
        5.5 * len(kinds), 3.6 * len(labels), rows=len(labels), cols=len(kinds)
    )
    for row, (reading, (title, unit)) in enumerate(labels.items()):
        for column, kind in enumerate(kinds):
            ax = axes[row, column]
            cell = curves[(curves["kind"] == kind) & (curves["reading"] == reading)]
            draw(ax, cell, reading)
            ax.set_xlabel("assets per portfolio", fontsize=8)
            _title(ax, f"{title} · {KIND_LABELS[kind]}", unit)
    return figure


def plot_portfolio_sizes(
    curves: pd.DataFrame, styles: Sequence[ArmStyle], directory: Path
) -> list[Path]:
    """Every portfolio reading over the portfolio size, one panel per kind.

    Rows are the readings of ``size_curves``, columns the portfolio kinds; each
    arm is a line through its estimates with a vertical line across each
    interval. The breach-rate panels shade the binomial band of a calibrated
    forecast and mark the nominal level.
    """

    def draw(ax: Axes, cell: pd.DataFrame, reading: str) -> None:
        _reading_panel(ax, cell, "size", reading, styles)

    figure = _size_grid(curves, READING_LABELS, draw)
    figure.suptitle(
        "Portfolio readings over the portfolio size: point and 95 % bootstrap "
        "interval over trials",
        color=INK,
        fontsize=10,
    )
    _arm_legend(figure, styles)
    return _save(figure, directory, "portfolio_sizes")


def plot_effect_sizes(
    curves: pd.DataFrame,
    styles: Sequence[EffectStyle],
    title: str,
    name: str,
    directory: Path,
) -> list[Path]:
    """Every portfolio reading's effects over the portfolio size.

    ``curves`` is ``effect_curves`` of the next day. The layout is that of
    ``plot_portfolio_sizes``, with the readings of ``EFFECT_READING_LABELS``
    as rows and each effect of ``styles`` as a line through its means with
    its intervals, against zero. ``title`` names the effects in the figure's
    heading, and the figure is saved as ``name``.
    """

    def draw(ax: Axes, cell: pd.DataFrame, reading: str) -> None:
        if reading in PERCENT_READINGS:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        _target_line(ax, 0.0)
        _curves(ax, cell, "size", styles)

    figure = _size_grid(curves, EFFECT_READING_LABELS, draw)
    figure.suptitle(
        f"{title} on the portfolio readings over the portfolio size: mean over "
        f"trials and 95 % bootstrap interval; 0 is no effect",
        color=INK,
        fontsize=10,
    )
    _effect_legend(figure, styles)
    return _save(figure, directory, name)


def plot_portfolio_horizons(
    curves: pd.DataFrame, styles: Sequence[ArmStyle], directory: Path
) -> list[Path]:
    """Every portfolio reading over the horizon, one figure per portfolio kind.

    ``curves`` stacks ``size_curves`` over the horizons (``stack_horizons``).
    Rows are the readings, columns the portfolio sizes; each arm is a line
    through its estimates at the horizons, drawn as in
    ``plot_portfolio_sizes``. The binomial band counts each origin once at
    every horizon, although the h-day outcomes of two origins less than h
    days apart overlap, so at long horizons it is narrower than the data
    support.
    """
    readings = list(READING_LABELS)
    files = []
    for kind in _kinds(curves):
        rows = curves[curves["kind"] == kind]
        sizes = sorted(rows["size"].unique())
        figure, axes = _figure(
            max(4.0 * len(sizes), MIN_HORIZON_FIGURE_WIDTH),
            3.4 * len(readings),
            rows=len(readings),
            cols=len(sizes),
        )
        for row, reading in enumerate(readings):
            title, unit = READING_LABELS[reading]
            for column, size in enumerate(sizes):
                ax = axes[row, column]
                cell = rows[(rows["size"] == size) & (rows["reading"] == reading)]
                _reading_panel(ax, cell, "horizon", reading, styles)
                ax.set_xlabel("trading days after the origin", fontsize=8)
                _title(ax, f"{title} · {size} assets", unit)
        figure.suptitle(
            f"Portfolio readings over the horizon, {KIND_LABELS[kind].lower()}: "
            f"point and 95 % interval over trials",
            color=INK,
            fontsize=10,
        )
        _arm_legend(figure, styles)
        files += _save(figure, directory, f"portfolio_horizons_{kind}")
    return files


def plot_portfolio_pit(
    pit: pd.DataFrame, styles: Sequence[ArmStyle], directory: Path
) -> list[Path]:
    """Mean PIT histogram per arm and size, one figure per portfolio kind."""
    centers = (np.arange(PIT_BINS) + 0.5) / PIT_BINS
    bins = [column for column in pit.columns if column.startswith("pit_bin_")]
    files = []
    for kind in _kinds(pit):
        rows = pit[pit["kind"] == kind]
        sizes = sorted(rows["size"].unique())
        figure, axes = _figure(
            2.2 * len(sizes), 2.0 * len(styles), rows=len(styles), cols=len(sizes)
        )
        for row, style in enumerate(styles):
            for column, size in enumerate(sizes):
                ax = axes[row, column]
                shares = rows[(rows["arm"] == style.name) & (rows["size"] == size)]
                ax.bar(
                    centers,
                    shares[bins].to_numpy()[0],
                    width=0.8 / PIT_BINS,
                    color=style.color if style.filled else SURFACE,
                    edgecolor=style.color,
                    linewidth=1,
                )
                _target_line(ax, 1 / PIT_BINS)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.5, 1])
                _title(ax, f"{size} assets", style.inline_label)
        figure.suptitle(
            f"Portfolio calibration, {KIND_LABELS[kind].lower()}: flat at "
            f"1/{PIT_BINS} is calibrated; U-shaped is too narrow",
            color=INK,
            fontsize=10,
        )
        files += _save(figure, directory, f"portfolio_pit_{kind}")
    return files

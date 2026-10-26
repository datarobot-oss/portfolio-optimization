# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared look of the simulator-comparison figures: palette, arm styles, panels.

Colors follow the simulator, so the CPU and GPU arm of one simulator share a
hue; the device is the second channel: filled marks are CPU, hollow marks GPU,
and every axis label names the arm. The ``no_fit`` reference is gray. The
per-arm figures draw the scenario simulators only (``simulator_styles``); each
measurement arm is drawn in the figure that reads it. The figures of the
dependence swap effects draw each effect in place of an arm
(``EFFECT_STYLES``), against zero, and so do the figure of the expected-return
attribution with each part (``ATTRIBUTION_STYLES``) and the figures of the
independence control with its one effect (``INDEPENDENCE_STYLES``).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator

from .analysis import INDEPENDENCE_EFFECT, REFERENCE_ARM
from .arms import DERIVED_DONORS, INDEPENDENT_ARM

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots 1-3 of the reference palette, validated all-pairs; the
# reference arm is gray because it is the baseline, not a contender. The swap
# cells take slots 4-6 and the historical-mean arm and the independence control
# slots 7-8, which pass only on adjacent pairs, so every figure names each arm
# on its axis or legend rather than by color alone.
SIMULATOR_COLORS = {
    "no_fit": MUTED,
    "gaussian": "#1baf7a",
    "kde": "#eb6834",
    "jointfm": "#2a78d6",
    "swap-jointfm-window": "#eda100",
    "swap-window-jointfm": "#e87ba4",
    "swap-window-window": "#008300",
    "jointfm-historical": "#6250d6",
    "window-independent": "#e34948",
}
SIMULATOR_ORDER = tuple(SIMULATOR_COLORS)
DEVICE_ORDER = ("CPU", "GPU")
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
class EffectStyle:
    """How one effect is drawn in place of an arm: a swap effect or an attribution part."""

    name: str
    color: str
    filled: bool
    definition: str

    @property
    def inline_label(self) -> str:
        return self.name


# The effects of ``analysis.swap_effects`` in drawing order. The two main
# effects and their interaction take categorical slots 1-3, validated
# all-pairs; the construction check, which should read zero, is a hollow
# dark gray that still separates from them under color-vision deficiency.
EFFECT_STYLES = (
    EffectStyle("marginals", "#2a78d6", True, "JointFM's marginals minus the window's"),
    EffectStyle(
        "dependence", "#eb6834", True, "JointFM's dependence minus the window's"
    ),
    EffectStyle(
        "interaction",
        "#1baf7a",
        True,
        "marginals' effect under JointFM's dependence minus under the window's",
    ),
    EffectStyle(
        "construction",
        INK_SECONDARY,
        False,
        "window corner minus no_fit, zero when the swap adds nothing",
    ),
)

# The parts of ``analysis.mean_attribution`` in drawing order, the total above
# the two parts that sum to it, in categorical slots 1-3.
ATTRIBUTION_STYLES = (
    EffectStyle("total", "#2a78d6", True, "jointfm minus no_fit"),
    EffectStyle(
        "expected_returns",
        "#eb6834",
        True,
        "jointfm minus jointfm-historical, JointFM's expected returns",
    ),
    EffectStyle(
        "scenarios",
        "#1baf7a",
        True,
        "jointfm-historical minus no_fit, JointFM's scenarios",
    ),
)

# The one effect of ``analysis.independence_effects``, in the control's color.
INDEPENDENCE_STYLES = (
    EffectStyle(
        INDEPENDENCE_EFFECT,
        SIMULATOR_COLORS[INDEPENDENT_ARM],
        True,
        f"{INDEPENDENT_ARM} minus no_fit, all dependence between assets removed",
    ),
)

# (table and figure name stem, heading, styles) of the swap effects and of the
# independence control, which both reports draw alike; the rebalancing report
# prefixes the stem with ``rebalancing_``.
SWAP_FIGURES = ("swap_effects", "Dependence swap effects", EFFECT_STYLES)
INDEPENDENCE_FIGURES = (
    "independence_control",
    f"Independence control ({INDEPENDENT_ARM} minus {REFERENCE_ARM})",
    INDEPENDENCE_STYLES,
)

# Either kind of drawn series: a simulator arm or an effect.
Style = ArmStyle | EffectStyle


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


def simulator_styles(per_trial: pd.DataFrame) -> list[ArmStyle]:
    """The scenario simulators' styles of ``arm_styles``, which the per-arm figures draw.

    The derived arms (``DERIVED_DONORS``) are measurement arms, each drawn
    only in the figure that reads it, so they neither crowd the simulators'
    panels nor set their axes.
    """
    return [
        style for style in arm_styles(per_trial) if style.name not in DERIVED_DONORS
    ]


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
    rows: Sequence[tuple[Style, float, float, float]],
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


# Spreads the arms at one position of a log axis apart, by this factor per arm.
ARM_OFFSET = 1.06


def _log_axis(ax, positions: Sequence[int]) -> None:
    """A log x axis with one labeled tick per position, such as sizes or horizons."""
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(positions, [str(position) for position in positions])
    ax.set_xlim(positions[0] / 1.4, positions[-1] * 1.4)
    ax.grid(axis="x", color=GRID, linewidth=1)


def _curves(ax, table: pd.DataFrame, x: str, styles: Sequence[Style]) -> None:
    """Each arm's ``estimate`` over the column ``x`` of ``table``, on a log axis.

    ``table`` holds one row per (``x``, ``arm``) with ``estimate``, ``ci_low``,
    and ``ci_high``; an effect stands in an ``arm`` row under its name. Each
    arm is a line through its estimates with a vertical line across each
    interval; the arms at one position are spread apart by ``ARM_OFFSET`` so
    their intervals do not overlap. Arms of ``styles`` that ``table`` lacks
    are skipped.
    """
    offsets = ARM_OFFSET ** (np.arange(len(styles)) - (len(styles) - 1) / 2)
    for offset, style in zip(offsets, styles):
        rows = table[table["arm"] == style.name].sort_values(x)
        if rows.empty:
            continue
        positions = rows[x].to_numpy() * offset
        ax.plot(
            positions,
            rows["estimate"],
            color=style.color,
            linewidth=LINE_WIDTH,
            linestyle="-" if style.filled else "--",
            zorder=3,
        )
        ax.vlines(
            positions, rows["ci_low"], rows["ci_high"], color=style.color, linewidth=1
        )
        ax.scatter(
            positions,
            rows["estimate"],
            s=MARKER_AREA,
            facecolor=style.color if style.filled else SURFACE,
            edgecolor=style.color,
            linewidth=MARKER_EDGE,
            zorder=4,
        )
    _log_axis(ax, sorted(table[x].unique()))


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


def _marker_legend(
    figure, entries: Sequence[tuple[str, bool, str]], ncols: int
) -> None:
    """A legend below the panels, one marker per (color, filled, label)."""
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=color if filled else SURFACE,
            markeredgecolor=color,
            markeredgewidth=MARKER_EDGE,
            label=label,
        )
        for color, filled, label in entries
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncols=ncols,
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
    )


def _arm_legend(figure, styles: Sequence[ArmStyle]) -> None:
    _marker_legend(
        figure,
        [(style.color, style.filled, style.inline_label) for style in styles],
        ncols=len(styles),
    )


# Effect definitions are too long for one legend row.
EFFECT_LEGEND_COLUMNS = 2


def _effect_legend(figure, styles: Sequence[EffectStyle]) -> None:
    """Each effect's marker with what it measures."""
    _marker_legend(
        figure,
        [
            (style.color, style.filled, f"{style.name}: {style.definition}")
            for style in styles
        ],
        ncols=EFFECT_LEGEND_COLUMNS,
    )


def _save(figure: Figure, directory: Path, name: str) -> list[Path]:
    paths = [directory / f"{name}.png", directory / f"{name}.svg"]
    figure.savefig(paths[0], dpi=FIGURE_DPI, facecolor=SURFACE)
    figure.savefig(paths[1], facecolor=SURFACE)
    return paths

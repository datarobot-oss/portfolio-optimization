# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correlation dilution line: each origin's correlation bias over its window's correlation.

The mean correlation bias (``corr_bias_trailing``) says how far an arm's
scenario correlations sit below the look-back window's, but not how the
shortfall arises. A straight-line fit of each origin's bias on the window's
mean pairwise correlation ρ̄ (``window_correlation`` of ``regimes``),
bias = intercept + slope · ρ̄, tells the shapes apart. Shrinking every
correlation toward a fixed prior ρ₀ with weight λ gives slope −λ and intercept
λ·ρ₀: an intercept near zero is a proportional dilution toward independence,
a positive one shrinkage toward a positive prior, and points that fall ever
further below the line at high ρ̄ a ceiling on the correlation the arm can
express, which the figure shows and the line does not. The forecast's
correlation rises ``forecast_slope`` = 1 + slope per unit of ρ̄; 1 keeps the
window's correlation.

The fit pools a cell's origins by least squares, and its intervals resample
trials as everywhere in the report: each resample sums its trials' origin
moments and refits, so a trial's origins stay together.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, CONFIDENCE, REFERENCE_ARM
from .plotting import (
    INK,
    INK_SECONDARY,
    LINE_WIDTH,
    MARKER_EDGE,
    SURFACE,
    ArmStyle,
    _arm_legend,
    _figure,
    _save,
    _target_line,
    _title,
)
from .regimes import regime_cells

BIAS = "corr_bias_trailing"
WINDOW_FEATURE = "window_correlation"
# Per-trial moments of the origins, in the order ``_line`` reads them.
MOMENTS = ("n", "x", "y", "xx", "xy")
# Many origins per panel, so small translucent points keep their density visible.
POINT_AREA = 9
POINT_ALPHA = 0.35
BAND_ALPHA = 0.25
BAND_POINTS = 50
DILUTION_COLUMNS = 3


def origin_points(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Each row's correlation bias beside its origin's window correlation.

    Parameters
    ----------
    frame : pd.DataFrame
        Results rows of one horizon.
    features : pd.DataFrame
        ``origin_features``, indexed by (trial, origin).

    Returns
    -------
    pd.DataFrame
        ``trial``, ``origin``, ``arm``, ``BIAS``, and ``WINDOW_FEATURE`` per row.

    Raises
    ------
    ValueError
        If ``frame`` holds more than one horizon, whose biases would mix in
        one line, or an origin has no features.
    """
    horizons = sorted(frame["horizon"].unique())
    if len(horizons) != 1:
        raise ValueError(f"A dilution line reads one horizon's rows; got {horizons}.")
    keys = pd.MultiIndex.from_frame(frame[["trial", "origin"]])
    window = features[WINDOW_FEATURE].reindex(keys)
    if window.isna().any():
        raise ValueError(f"Some origins have no {WINDOW_FEATURE}.")
    return frame[["trial", "origin", "arm", BIAS]].assign(
        **{WINDOW_FEATURE: window.to_numpy()}
    )


def _line(moments: np.ndarray, axis: int) -> np.ndarray:
    """Least-squares (intercept, slope) of the moments summed along ``axis``."""
    n, x, y, xx, xy = np.moveaxis(moments.sum(axis=axis), -1, 0)
    # Sums over one window correlation divide rounding noise; fit_line masks them.
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n * xy - x * y) / (n * xx - x * x)
    return np.stack([(y - slope * x) / n, slope], axis=-1)


def fit_line(points: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    """The dilution line of one arm's ``origin_points`` and its trial resamples.

    Parameters
    ----------
    points : pd.DataFrame
        ``origin_points`` of one arm.
    name : str
        What the points are, such as the arm and regime, for the errors.

    Returns
    -------
    tuple
        (intercept, slope) over all origins, and ``BOOTSTRAP_RESAMPLES`` rows
        of (intercept, slope), each refitted on trials resampled with
        replacement. A resample that draws a single trial holding a single
        origin, possible only among few trials, has no line, and its row is NaN.

    Raises
    ------
    ValueError
        If the points hold one window correlation only, which fits no line.
    """
    x = points[WINDOW_FEATURE].to_numpy()
    y = points[BIAS].to_numpy()
    if np.ptp(x) == 0:
        raise ValueError(
            f"The origins of {name} share one {WINDOW_FEATURE}, so no line can "
            f"be fitted."
        )
    trials = pd.DataFrame(
        dict(zip(MOMENTS, (np.ones_like(x), x, y, x * x, x * y)))
    ).groupby(points["trial"].to_numpy())
    moments = trials.sum().to_numpy()
    lowest = trials["x"].min().to_numpy()
    highest = trials["x"].max().to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resamples = rng.integers(0, len(moments), size=(BOOTSTRAP_RESAMPLES, len(moments)))
    lines = _line(moments[resamples], axis=1)
    single = lowest[resamples].min(axis=1) == highest[resamples].max(axis=1)
    lines[single] = np.nan
    return _line(moments, axis=0), lines


def _interval(resamples: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Percentile bounds of each column of ``resamples``, and its undefined rows.

    The bounds are NaN when any resample has no line, because dropping those
    resamples would narrow the interval exactly where trials are too few.
    """
    undefined = int(np.isnan(resamples).any(axis=1).sum())
    if undefined:
        nan = np.full(resamples.shape[1:], np.nan)
        return nan, nan, undefined
    tail = (1 - CONFIDENCE) / 2
    low, high = np.quantile(resamples, [tail, 1 - tail], axis=0)
    return low, high, undefined


def dilution_lines(
    frame: pd.DataFrame, features: pd.DataFrame, regimes: pd.DataFrame
) -> pd.DataFrame:
    """Every arm's dilution line over all origins and in every regime.

    Parameters
    ----------
    frame : pd.DataFrame
        Results rows of one horizon.
    features, regimes : pd.DataFrame
        ``origin_features`` and the regimes of ``terciles``.

    Returns
    -------
    pd.DataFrame
        One row per (feature, regime, arm), all origins first with ``OVERALL``
        in both: its numbers of trials and origins, and the ``intercept``,
        ``slope``, and ``forecast_slope`` (1 + slope), each with its bootstrap
        interval (``_low``, ``_high``). ``undefined_resamples`` counts the
        resamples without a line; where it is above 0 the intervals are NaN.

    Raises
    ------
    ValueError
        As ``origin_points``, ``regime_cells``, and ``fit_line``.
    """
    points = origin_points(frame, features)
    rows = []
    for feature, regime, mask in regime_cells(frame, regimes):
        for arm, cell in points[mask].groupby("arm", sort=False):
            (intercept, slope), resamples = fit_line(
                cell, f"{arm} in the {regime} {feature} regime"
            )
            low, high, undefined = _interval(resamples)
            rows.append(
                {
                    "feature": feature,
                    "regime": regime,
                    "arm": arm,
                    "trials": cell["trial"].nunique(),
                    "origins": len(cell),
                    "undefined_resamples": undefined,
                    "intercept": intercept,
                    "intercept_low": low[0],
                    "intercept_high": high[0],
                    "slope": slope,
                    "slope_low": low[1],
                    "slope_high": high[1],
                    "forecast_slope": 1 + slope,
                    "forecast_slope_low": 1 + low[1],
                    "forecast_slope_high": 1 + high[1],
                }
            )
    return pd.DataFrame(rows)


def plot_dilution(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    styles: Sequence[ArmStyle],
    directory: Path,
) -> list[Path]:
    """Each origin's correlation bias over its window correlation, one panel per arm.

    Draws every arm of ``styles`` but the reference, whose next-day bias is
    zero by construction: its origins as points, its dilution line over all
    origins, and the line's 95 % bootstrap band, against the zero bias of a
    forecast that keeps the window's correlation. A panel whose resamples
    include one without a line has no band and says so.
    """
    points = origin_points(frame, features)
    contenders = [style for style in styles if style.name != REFERENCE_ARM]
    cols = min(len(contenders), DILUTION_COLUMNS)
    rows = -(-len(contenders) // cols)
    figure, axes = _figure(4.6 * cols, 4.2 * rows, rows=rows, cols=cols)
    window = points[WINDOW_FEATURE]
    grid = np.linspace(window.min(), window.max(), BAND_POINTS)
    for ax, style in zip(axes.flat, contenders):
        cell = points[points["arm"] == style.name]
        (intercept, slope), resamples = fit_line(cell, style.name)
        low, high, undefined = _interval(resamples[:, :1] + resamples[:, 1:] * grid)
        _target_line(ax, 0.0)
        ax.scatter(
            cell[WINDOW_FEATURE],
            cell[BIAS],
            s=POINT_AREA,
            facecolor=style.color if style.filled else SURFACE,
            edgecolor=style.color,
            linewidth=MARKER_EDGE / 2,
            alpha=POINT_ALPHA,
            zorder=3,
        )
        ax.fill_between(
            grid, low, high, color=style.color, alpha=BAND_ALPHA, linewidth=0, zorder=4
        )
        ax.plot(
            grid, intercept + slope * grid, color=INK, linewidth=LINE_WIDTH, zorder=5
        )
        ax.set_xlabel(
            "look-back window's mean pairwise correlation",
            color=INK_SECONDARY,
            fontsize=8,
        )
        ax.set_ylabel(
            "correlation bias, median signed difference",
            color=INK_SECONDARY,
            fontsize=8,
        )
        _title(
            ax,
            style.inline_label,
            f"bias = {intercept:+.3f} {slope:+.3f} × window; "
            f"forecast rises {1 + slope:.2f} per unit"
            + (f"; no band, {undefined} resamples without a line" if undefined else ""),
        )
    for ax in list(axes.flat)[len(contenders) :]:
        ax.set_visible(False)
    figure.suptitle(
        "Correlation dilution: each origin's bias over its window's correlation, "
        "with the fitted line and its 95 % bootstrap band over trials",
        color=INK,
        fontsize=10,
    )
    _arm_legend(figure, contenders)
    return _save(figure, directory, "dilution")

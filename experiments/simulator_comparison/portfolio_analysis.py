# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portfolio readings per arm as curves over the portfolio size.

The portfolio size, the number of assets a portfolio holds, is a reporting
dimension of its own: an equal-volatility portfolio's variance scales with
``(1 - rho) / N + rho`` for ``N`` assets of mean pairwise correlation
``rho``, so size 1 reads the per-asset distributions alone and larger sizes
read the dependence ever more. Every reading starts from ``aggregate``'s
per-trial means of a cell's columns (``PortfolioScores``), and its interval
resamples trials, as everywhere in the report.
"""

import numpy as np
import pandas as pd
from scipy.stats import binom

from .analysis import (
    BOOTSTRAP_SEED,
    CONFIDENCE,
    PIT_COLUMNS,
    REFERENCE_ARM,
    bootstrap_interval,
)
from .scores import (
    PORTFOLIO_READINGS,
    PORTFOLIO_VAR_LEVELS,
    cell_column,
    recorded_cells,
)

SPREAD_TARGET = 1.0
CURVE_COLUMNS = [
    "kind",
    "size",
    "arm",
    "reading",
    "estimate",
    "ci_low",
    "ci_high",
    "target",
    "band_low",
    "band_high",
]


def breach_reading(level: float) -> str:
    """Name of the breach-rate reading at VaR ``level``."""
    return f"breach_{round(100 * level):02d}"


def breach_band(level: float, origins: int) -> tuple[float, float]:
    """Central ``CONFIDENCE`` range of a calibrated forecast's breach rate.

    A calibrated forecast breaches its VaR at ``level`` on each day with
    probability ``level``, so over ``origins`` independent days its breach
    count is binomial. Each origin counts once although several portfolios
    are scored on it: they share the day, so their breaches are not
    independent, and counting them separately would narrow the band below
    what the data support.
    """
    tail = (1 - CONFIDENCE) / 2
    low, high = binom.ppf([tail, 1 - tail], origins, level)
    return float(low / origins), float(high / origins)


def _spread_ratio(z2: np.ndarray, axis=None) -> np.ndarray:
    """Scenario over realized spread: the inverse root of the mean ``z2``."""
    return 1 / np.sqrt(np.mean(z2, axis=axis))


def _cell(per_trial: pd.DataFrame, kind: str, size: int, reading: str) -> pd.DataFrame:
    return per_trial[cell_column(kind, size, reading)].unstack("arm")


def _row(kind, size, arm, reading, estimate, interval, target, band=(None, None)):
    return dict(
        zip(
            CURVE_COLUMNS,
            (kind, size, arm, reading, estimate, *interval, target, *band),
        )
    )


def size_curves(per_trial: pd.DataFrame) -> pd.DataFrame:
    """Every portfolio reading per (kind, size, arm), with its interval.

    Readings:

    - ``breach_05``, ``breach_01``: mean breach rate over trials, with the
      binomial band of a calibrated forecast over the arm's origins
      (``breach_band``) as ``band_low`` and ``band_high``; inside the band
      is calibrated.
    - ``spread_ratio``: scenario over realized spread over all trials,
      ``1 / sqrt(mean z2)``, against the target 1; below 1 is too narrow.
    - ``crps_skill``: median over trials of the percent reduction of the
      cell's CRPS against ``REFERENCE_ARM``, for every other arm; above 0
      beats the reference.

    Raises
    ------
    KeyError
        If ``per_trial`` has no ``REFERENCE_ARM``.
    """
    origins = per_trial["origins"].groupby("arm").sum()
    rows = []
    for kind, size in recorded_cells(per_trial.columns):
        for level in PORTFOLIO_VAR_LEVELS:
            reading = breach_reading(level)
            for arm, values in _cell(per_trial, kind, size, reading).items():
                values = values.dropna().to_numpy()
                interval = bootstrap_interval(
                    values, np.mean, np.random.default_rng(BOOTSTRAP_SEED)
                )
                band = breach_band(level, int(origins[arm]))
                rows.append(
                    _row(kind, size, arm, reading, values.mean(), interval, level, band)
                )
        for arm, values in _cell(per_trial, kind, size, "z2").items():
            values = values.dropna().to_numpy()
            interval = bootstrap_interval(
                values, _spread_ratio, np.random.default_rng(BOOTSTRAP_SEED)
            )
            rows.append(
                _row(
                    kind,
                    size,
                    arm,
                    "spread_ratio",
                    float(_spread_ratio(values)),
                    interval,
                    SPREAD_TARGET,
                )
            )
        crps = _cell(per_trial, kind, size, "crps")
        skill = 100 * (1 - crps.div(crps[REFERENCE_ARM], axis=0))
        for arm, values in skill.drop(columns=REFERENCE_ARM).items():
            values = values.dropna().to_numpy()
            interval = bootstrap_interval(
                values, np.median, np.random.default_rng(BOOTSTRAP_SEED)
            )
            rows.append(
                _row(
                    kind,
                    size,
                    arm,
                    "crps_skill",
                    float(np.median(values)),
                    interval,
                    0.0,
                )
            )
    return pd.DataFrame(rows, columns=CURVE_COLUMNS)


def effect_curves(effects: pd.DataFrame) -> pd.DataFrame:
    """The portfolio rows of ``swap_effects`` as curves over the portfolio size.

    Returns
    -------
    pd.DataFrame
        One row per (kind, size, effect, reading) of the cells' own readings,
        such as ``breach_05``, with the effect's mean over trials as
        ``estimate`` and its interval. The effect is named in ``arm``, where
        ``plotting._curves`` reads the series.
    """
    names = {
        cell_column(kind, size, reading): (kind, size, reading)
        for kind, size in recorded_cells(effects["metric"].unique())
        for reading in PORTFOLIO_READINGS
    }
    rows = effects[effects["metric"].isin(names)]
    cells = pd.DataFrame(
        rows["metric"].map(names).tolist(),
        columns=["kind", "size", "reading"],
        index=rows.index,
    )
    return cells.join(
        rows[["effect", "mean", "ci_low", "ci_high"]].rename(
            columns={"effect": "arm", "mean": "estimate"}
        )
    )


def portfolio_pit(per_trial: pd.DataFrame) -> pd.DataFrame:
    """Mean PIT histogram per (kind, size, arm), one column per bin."""
    tables = []
    for kind, size in recorded_cells(per_trial.columns):
        columns = [cell_column(kind, size, name) for name in PIT_COLUMNS]
        pit = per_trial[columns].groupby("arm").mean()
        pit.columns = list(PIT_COLUMNS)
        tables.append(pit.reset_index().assign(kind=kind, size=size))
    return pd.concat(tables, ignore_index=True)[["kind", "size", "arm", *PIT_COLUMNS]]

# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scores of one arm's scenarios against the realized returns over one horizon.

Every score treats the scenario set as the discrete, equally weighted
distribution the optimizer receives, so it measures what the optimizer sees.
Returns are log returns over the horizon's ``h`` trading days after the
origin (``horizons``). CRPS, the energy score, and the variogram score are
taken in units of each asset's trailing realized daily volatility times
``sqrt(h)``, so calm and volatile assets weigh alike in the average over
assets and the joint scores, and horizons compare on one scale; portfolio
scores in units of each portfolio's own trailing volatility times ``sqrt(h)``.
Skill against ``no_fit`` needs that arm's row at the same origin and horizon,
so it is computed from the results table, not here.
"""

import re
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.spatial.distance import pdist

from .config import CvarConfig
from .portfolios import (
    PORTFOLIO_STREAM,
    draw_portfolios,
    portfolio_cells,
    portfolio_log_returns,
)
from .trials import OriginInputs

TRAILING_STD_WINDOW = 63
INTERVAL_LEVELS = (0.50, 0.90)
VAR_LEVEL = 0.05
# Bin edges every 0.05, so the central intervals at 10 %, 20 %, ..., 90 % of
# a reliability curve fall on bin edges.
PIT_BINS = 20
# Bounds the energy score's pairwise term, which is quadratic in scenarios.
ENERGY_SUBSAMPLE = 1000
# Order 0.5 discriminated dependence best with the least sampling noise in
# Scheuerer & Hamill (2015), https://doi.org/10.1175/MWR-D-14-00269.1.
VARIOGRAM_ORDER = 0.5
# Scenarios entering the variogram expectation; its cost is scenarios x pairs.
VARIOGRAM_SUBSAMPLE = 5000
# Scenarios per block of the variogram expectation, which bounds its memory.
VARIOGRAM_BLOCK = 500
PORTFOLIO_VAR_LEVELS = (0.05, 0.01)
# Asset subsets per portfolio size and origin.
PORTFOLIO_SUBSETS = 5
PORTFOLIO_CRPS = "portfolio_crps"


def sample_crps(samples: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """CRPS of each column's empirical distribution at its realized value.

    Uses ``E|X - y| - E|X - X'| / 2`` with the pairwise term in its sorted
    form, which is exact for the empirical distribution of ``samples``.

    Parameters
    ----------
    samples : np.ndarray
        Scenarios, shape ``(S, D)``.
    realized : np.ndarray
        Outcomes, shape ``(D,)``.

    Returns
    -------
    np.ndarray
        CRPS per column, shape ``(D,)``.
    """
    ordered = np.sort(samples, axis=0)
    count = ordered.shape[0]
    weights = (2 * np.arange(1, count + 1) - count - 1)[:, None]
    accuracy = np.abs(ordered - realized).mean(axis=0)
    return accuracy - (weights * ordered).sum(axis=0) / count**2


def energy_score(
    samples: np.ndarray,
    realized: np.ndarray,
    *,
    subsample: int,
    rng: np.random.Generator,
) -> float:
    """Energy score of the empirical distribution of ``samples`` at ``realized``.

    ``E||X - y|| - E||X - X'|| / 2`` (Gneiting & Raftery 2007,
    https://doi.org/10.1198/016214506000001437), a proper score of the joint
    distribution that rewards correct dependence between columns. The first
    term uses every scenario. The pairwise term over all ``S`` scenarios is
    ``(S - 1) / S`` times the mean distance over distinct pairs; that mean is
    estimated without bias from distinct pairs of at most ``subsample``
    scenarios drawn without replacement, and is exact when ``S <= subsample``.

    Parameters
    ----------
    samples : np.ndarray
        Scenarios, shape ``(S, D)``, ``S >= 2``.
    realized : np.ndarray
        Outcome, shape ``(D,)``.
    subsample : int
        Largest number of scenarios entering the pairwise term.
    rng : np.random.Generator
        Draws the subsample.

    Returns
    -------
    float
    """
    count = samples.shape[0]
    accuracy = np.linalg.norm(samples - realized, axis=1).mean()
    chosen = samples
    if count > subsample:
        chosen = samples[rng.choice(count, subsample, replace=False)]
    spread = (count - 1) / count * pdist(chosen).mean()
    return float(accuracy - spread / 2)


def variogram_score(
    samples: np.ndarray,
    realized: np.ndarray,
    *,
    order: float,
    subsample: int,
    rng: np.random.Generator,
) -> float:
    """Unweighted variogram score of order ``order``, mean over column pairs.

    ``mean_{i<j} (|y_i - y_j|^p - E|X_i - X_j|^p)^2`` (Scheuerer & Hamill
    2015, https://doi.org/10.1175/MWR-D-14-00269.1), a proper score of the
    joint distribution that compares every pair's expected absolute
    difference with the realized one, and so reacts to wrong correlations
    that the energy score barely sees. Assets have no natural distance, so
    every pair weighs alike. The expectation uses at most ``subsample``
    scenarios drawn without replacement and is exact for the empirical
    distribution when ``S <= subsample``; its estimation noise adds a bias of
    its variance, about ``1 / subsample`` of the pair's own variance.

    Parameters
    ----------
    samples : np.ndarray
        Scenarios, shape ``(S, D)``, ``D >= 2``.
    realized : np.ndarray
        Outcome, shape ``(D,)``.
    order : float
        Power ``p`` of the absolute differences.
    subsample : int
        Largest number of scenarios entering the expectation.
    rng : np.random.Generator
        Draws the subsample.

    Returns
    -------
    float
    """
    first, second = np.triu_indices(samples.shape[1], 1)
    chosen = samples
    if samples.shape[0] > subsample:
        chosen = samples[rng.choice(samples.shape[0], subsample, replace=False)]
    expected = np.zeros(len(first))
    for start in range(0, len(chosen), VARIOGRAM_BLOCK):
        block = chosen[start : start + VARIOGRAM_BLOCK]
        expected += (np.abs(block[:, first] - block[:, second]) ** order).sum(axis=0)
    expected /= len(chosen)
    observed = np.abs(realized[first] - realized[second]) ** order
    return float(np.mean((observed - expected) ** 2))


def probability_integral_transform(
    samples: np.ndarray, realized: np.ndarray
) -> np.ndarray:
    """Share of each column's scenarios below its outcome, ties counted half."""
    below = (samples < realized).mean(axis=0)
    tied = (samples == realized).mean(axis=0)
    return below + tied / 2


def interval_coverage(
    samples: np.ndarray, realized: np.ndarray, level: float
) -> np.ndarray:
    """Whether each outcome lies in its column's central ``level`` interval."""
    low, high = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return (realized >= low) & (realized <= high)


def _off_diagonal(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], 1)]


def pairwise_correlations(returns: np.ndarray, label: str) -> np.ndarray:
    """Correlation of every pair of columns of ``returns`` (rows x columns).

    Raises
    ------
    ValueError
        If a correlation is undefined because a column is constant; ``label``
        names the returns in the message.
    """
    correlations = _off_diagonal(np.corrcoef(returns.T))
    if not np.isfinite(correlations).all():
        raise ValueError(
            f"The {label} correlations are not finite: a column is constant."
        )
    return correlations


def expected_return_correlation(
    expected_returns: np.ndarray, realized: np.ndarray
) -> float:
    """Pearson correlation across assets of expected returns with realized returns.

    Positive when the assets the optimizer expects to gain more do gain more;
    zero for expected returns that carry no information about the outcome.

    Raises
    ------
    ValueError
        If the two vectors' shapes differ, or either is constant, which
        leaves the correlation undefined.
    """
    if expected_returns.shape != realized.shape:
        raise ValueError(
            f"Expected returns of shape {expected_returns.shape} cannot be "
            f"compared with realized returns of shape {realized.shape}."
        )
    if np.ptp(expected_returns) == 0 or np.ptp(realized) == 0:
        raise ValueError(
            "The expected-return correlation is undefined: the expected or the "
            "realized returns are equal across assets."
        )
    return float(np.corrcoef(expected_returns, realized)[0, 1])


def _pit_column(index: int) -> str:
    return f"pit_bin_{index:02d}"


def _cover_column(level: float) -> str:
    return f"cover_{round(100 * level)}"


class ScenarioScores:
    """Readings of the Return-input backtest, plus PIT and energy score, per row.

    Every reading compares the scenarios of the log return over one horizon of
    ``h`` days with the realized ``h``-day log return. The *horizon scale* of
    an asset is its trailing realized daily volatility times ``sqrt(h)``.
    Columns:

    - ``crps``: sample CRPS per asset in horizon-scale units, mean over assets.
    - ``energy_score``: energy score of the asset vector in horizon-scale
      units.
    - ``variogram_score``: variogram score of order ``VARIOGRAM_ORDER`` of the
      asset vector in horizon-scale units.
    - ``spread``: scenario standard deviation over the horizon scale, median
      over assets; ~1 for a well-scaled forecast of independent days.
    - ``cover_50``, ``cover_90``: share of assets whose outcome lies in the
      central 50 % and 90 % scenario interval.
    - ``pit_bin_00`` .. ``pit_bin_19``: share of assets whose PIT falls into
      each of ``PIT_BINS`` equal bins of [0, 1]; flat for a calibrated forecast.
    - ``corr_error_trailing``, ``corr_bias_trailing``: median absolute and
      signed difference of the pairwise scenario correlations from those of the
      look-back window's daily returns. Zero by construction for ``no_fit`` at
      the next day.
    - ``corr_error_forward``: median absolute difference from the correlations
      of the daily returns after the origin. Both references are daily
      correlations at every horizon, which equal the h-day correlations of
      independent days; 63 forward days cannot estimate an h-day correlation.
    - ``ew_var_exceeded``: 1 when the equal-weight portfolio's realized h-day
      log return falls below its scenarios' ``VAR_LEVEL`` quantile; ~0.05 on
      average.
    - ``mean_correlation``: correlation across assets of the arm's expected
      returns, the next-day ``CvarData.mean`` its rebalancing runs optimize
      on, with the realized h-day log returns
      (``expected_return_correlation``). It scores the expected returns, which
      no other reading sees; every arm on the window's mean reads the same.
    """

    columns = (
        "crps",
        "energy_score",
        "variogram_score",
        "spread",
        *(_cover_column(level) for level in INTERVAL_LEVELS),
        *(_pit_column(i) for i in range(PIT_BINS)),
        "corr_error_trailing",
        "corr_bias_trailing",
        "corr_error_forward",
        "ew_var_exceeded",
        "mean_correlation",
    )
    parameters = {
        "trailing_std_window": TRAILING_STD_WINDOW,
        "interval_levels": list(INTERVAL_LEVELS),
        "var_level": VAR_LEVEL,
        "pit_bins": PIT_BINS,
        "energy_subsample": ENERGY_SUBSAMPLE,
        "variogram_order": VARIOGRAM_ORDER,
        "variogram_subsample": VARIOGRAM_SUBSAMPLE,
    }

    def __call__(
        self,
        inputs: OriginInputs,
        horizon: int,
        samples: np.ndarray,
        expected_returns: np.ndarray,
    ) -> Mapping[str, float]:
        """Score ``horizon``-day ``samples`` (scenarios x assets) at one origin.

        Raises
        ------
        ValueError
            If the look-back window is shorter than ``TRAILING_STD_WINDOW``,
            the forward window holds fewer than two returns or fewer than
            ``horizon``, or a correlation is undefined because a column is
            constant, or as ``expected_return_correlation`` raises.
        """
        window = inputs.returns_dict["returns"].to_numpy()
        forward = inputs.forward.to_numpy()
        if len(window) < TRAILING_STD_WINDOW:
            raise ValueError(
                f"The look-back window holds {len(window)} returns; the scores "
                f"need {TRAILING_STD_WINDOW} for the trailing volatility."
            )
        if len(forward) < 2:
            raise ValueError("The forward correlations need at least two returns.")
        realized = inputs.realized(horizon).to_numpy()
        scale = window[-TRAILING_STD_WINDOW:].std(axis=0) * np.sqrt(horizon)

        scores = {
            "crps": float((sample_crps(samples, realized) / scale).mean()),
            "energy_score": energy_score(
                samples / scale,
                realized / scale,
                subsample=ENERGY_SUBSAMPLE,
                rng=np.random.default_rng(inputs.seed),
            ),
            "variogram_score": variogram_score(
                samples / scale,
                realized / scale,
                order=VARIOGRAM_ORDER,
                subsample=VARIOGRAM_SUBSAMPLE,
                rng=np.random.default_rng(inputs.seed),
            ),
            "spread": float(np.median(samples.std(axis=0) / scale)),
        }
        for level in INTERVAL_LEVELS:
            scores[_cover_column(level)] = float(
                interval_coverage(samples, realized, level).mean()
            )
        counts, _ = np.histogram(
            probability_integral_transform(samples, realized),
            bins=PIT_BINS,
            range=(0.0, 1.0),
        )
        for index, count in enumerate(counts):
            scores[_pit_column(index)] = float(count / samples.shape[1])

        scenario = pairwise_correlations(samples, "scenario")
        trailing = pairwise_correlations(window, "look-back window")
        scores["corr_error_trailing"] = float(np.median(np.abs(scenario - trailing)))
        scores["corr_bias_trailing"] = float(np.median(scenario - trailing))
        scores["corr_error_forward"] = float(
            np.median(np.abs(scenario - pairwise_correlations(forward, "forward")))
        )

        weights = np.full(samples.shape[1], 1.0 / samples.shape[1])
        portfolio = np.log(np.exp(samples) @ weights)
        realized_portfolio = np.log(np.exp(realized) @ weights)
        scores["ew_var_exceeded"] = float(
            realized_portfolio < np.quantile(portfolio, VAR_LEVEL)
        )
        scores["mean_correlation"] = expected_return_correlation(
            np.asarray(expected_returns), realized
        )
        return scores


PORTFOLIO_READINGS = (
    *(f"breach_{round(100 * level):02d}" for level in PORTFOLIO_VAR_LEVELS),
    "crps",
    "z2",
    *(_pit_column(i) for i in range(PIT_BINS)),
)
_CELL_COLUMN = re.compile(r"^portfolio_(?P<kind>[a-z_]+?)_(?P<size>\d+)_crps$")


def cell_column(kind: str, size: int, reading: str) -> str:
    """Results column of one reading of one (kind, size) portfolio cell."""
    return f"portfolio_{kind}_{size}_{reading}"


def recorded_cells(columns: Sequence[str]) -> list[tuple[str, int]]:
    """The (kind, size) cells whose columns ``columns`` holds, in column order."""
    return [
        (match["kind"], int(match["size"]))
        for match in map(_CELL_COLUMN.match, columns)
        if match
    ]


class PortfolioScores:
    """Readings of random portfolios of the trial's assets, per row.

    At every origin, ``draw_portfolios`` draws ``PORTFOLIO_SUBSETS`` asset
    subsets per size and one portfolio per kind on each, from the origin's
    scenario seed, so every arm and every horizon scores the same portfolios.
    A portfolio's return over a horizon of ``h`` days holds its weights from
    the origin on, without rebalancing. Each reading is the mean over a
    cell's subsets:

    - ``portfolio_<kind>_<size>_breach_05``, ``..._breach_01``: share of
      portfolios whose realized h-day log return falls below their scenarios'
      5 % and 1 % quantile, the value at risk; the level on average.
    - ``..._crps``: sample CRPS in units of the portfolio's trailing realized
      daily volatility times ``sqrt(h)``.
    - ``..._z2``: squared realized return minus the scenario mean, over the
      scenario variance; the report turns its mean over a trial into the
      ratio of scenario to realized spread, ``1 / sqrt(mean z2)``.
    - ``..._pit_bin_00`` .. ``..._pit_bin_19``: share of portfolios whose
      PIT falls into each of ``PIT_BINS`` equal bins of [0, 1].

    ``portfolio_crps`` is the CRPS averaged over every cell, the portfolio
    score the report ranks the arms by.

    Parameters
    ----------
    sizes : sequence of int
        Portfolio sizes, strictly increasing.
    bounds : CvarConfig
        Weight, cash, and leverage bounds of the random portfolios.
    n_assets : int
        Assets per trial universe.

    Raises
    ------
    ValueError
        If a size exceeds ``n_assets``.
    """

    def __init__(self, sizes: Sequence[int], bounds: CvarConfig, n_assets: int):
        if max(sizes) > n_assets:
            raise ValueError(
                f"Portfolio sizes {list(sizes)} exceed the {n_assets} assets of a "
                f"trial."
            )
        self.bounds = bounds
        self.cells = portfolio_cells(sizes, bounds)
        self.columns = (
            PORTFOLIO_CRPS,
            *(
                cell_column(kind, size, reading)
                for kind, size in self.cells
                for reading in PORTFOLIO_READINGS
            ),
        )
        self.parameters = {
            "sizes": list(sizes),
            "cells": [list(cell) for cell in self.cells],
            "subsets": PORTFOLIO_SUBSETS,
            "var_levels": list(PORTFOLIO_VAR_LEVELS),
            "pit_bins": PIT_BINS,
            "trailing_std_window": TRAILING_STD_WINDOW,
            "bounds": bounds.model_dump(
                include={"w_min", "w_max", "c_min", "c_max", "L_tar"}
            ),
        }

    def __call__(
        self,
        inputs: OriginInputs,
        horizon: int,
        samples: np.ndarray,
        expected_returns: np.ndarray,
    ) -> Mapping[str, float]:
        """Score every portfolio cell on ``horizon``-day ``samples`` (scenarios x assets).

        The random portfolios do not depend on ``expected_returns``, which no
        reading here uses.

        Raises
        ------
        ValueError
            If the look-back window is shorter than ``TRAILING_STD_WINDOW``,
            the forward window holds fewer than ``horizon`` returns, or a
            portfolio loses its whole value in a scenario.
        """
        window = inputs.returns_dict["returns"].to_numpy()
        if len(window) < TRAILING_STD_WINDOW:
            raise ValueError(
                f"The look-back window holds {len(window)} returns; the scores "
                f"need {TRAILING_STD_WINDOW} for the trailing volatility."
            )
        weights = draw_portfolios(
            np.random.default_rng([inputs.seed, PORTFOLIO_STREAM]),
            samples.shape[1],
            self.cells,
            PORTFOLIO_SUBSETS,
            self.bounds,
        )
        scenarios = portfolio_log_returns(samples, weights)
        realized = portfolio_log_returns(
            inputs.realized(horizon).to_numpy()[None, :], weights
        )[:, 0]
        scale = portfolio_log_returns(window[-TRAILING_STD_WINDOW:], weights).std(
            axis=1
        ) * np.sqrt(horizon)

        readings = {
            f"breach_{round(100 * level):02d}": realized < quantile
            for level, quantile in zip(
                PORTFOLIO_VAR_LEVELS,
                np.quantile(scenarios, PORTFOLIO_VAR_LEVELS, axis=1),
            )
        }
        readings["crps"] = sample_crps(scenarios.T, realized) / scale
        readings["z2"] = (
            (realized - scenarios.mean(axis=1)) / scenarios.std(axis=1)
        ) ** 2
        pit = probability_integral_transform(scenarios.T, realized)
        bins = np.minimum((pit * PIT_BINS).astype(int), PIT_BINS - 1)
        for index in range(PIT_BINS):
            readings[_pit_column(index)] = bins == index

        scores = {PORTFOLIO_CRPS: float(readings["crps"].mean())}
        for position, (kind, size) in enumerate(self.cells):
            rows = slice(
                position * PORTFOLIO_SUBSETS, (position + 1) * PORTFOLIO_SUBSETS
            )
            for reading in PORTFOLIO_READINGS:
                scores[cell_column(kind, size, reading)] = float(
                    np.mean(readings[reading][rows])
                )
        return scores

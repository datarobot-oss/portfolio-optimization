# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scores of one arm's scenarios against the realized next-day returns.

Every score treats the scenario set as the discrete, equally weighted
distribution the optimizer receives, so it measures what the optimizer sees.
Returns are log returns. CRPS and the energy score are taken in units of each
asset's trailing realized volatility, so calm and volatile assets weigh alike
in the average over assets and the joint score. Skill against ``no_fit`` needs
that arm's row at the same origin, so it is computed from the results table,
not here.
"""

from collections.abc import Mapping

import numpy as np
from scipy.spatial.distance import pdist

from portfolio_optimization.cvar_data import CvarData

from .trials import OriginInputs

TRAILING_STD_WINDOW = 63
INTERVAL_LEVELS = (0.50, 0.90)
VAR_LEVEL = 0.05
# Bin edges every 0.05, so the central intervals at 10 %, 20 %, ..., 90 % of
# a reliability curve fall on bin edges.
PIT_BINS = 20
# Bounds the energy score's pairwise term, which is quadratic in scenarios.
ENERGY_SUBSAMPLE = 1000


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


def _pit_column(index: int) -> str:
    return f"pit_bin_{index:02d}"


def _cover_column(level: float) -> str:
    return f"cover_{round(100 * level)}"


class ScenarioScores:
    """Readings of the Return-input backtest, plus PIT and energy score, per row.

    Columns:

    - ``crps``: sample CRPS per asset in trailing-volatility units, mean over
      assets.
    - ``energy_score``: energy score of the asset vector in trailing-volatility
      units.
    - ``spread``: scenario standard deviation over trailing realized
      volatility, median over assets; ~1 for a well-scaled forecast.
    - ``cover_50``, ``cover_90``: share of assets whose outcome lies in the
      central 50 % and 90 % scenario interval.
    - ``pit_bin_00`` .. ``pit_bin_19``: share of assets whose PIT falls into
      each of ``PIT_BINS`` equal bins of [0, 1]; flat for a calibrated forecast.
    - ``corr_error_trailing``, ``corr_bias_trailing``: median absolute and
      signed difference of the pairwise scenario correlations from those of the
      look-back window. Zero by construction for ``no_fit``.
    - ``corr_error_forward``: median absolute difference from the correlations
      of the returns after the origin.
    - ``ew_var_exceeded``: 1 when the equal-weight portfolio's realized log
      return falls below its scenarios' ``VAR_LEVEL`` quantile; ~0.05 on
      average.
    """

    columns = (
        "crps",
        "energy_score",
        "spread",
        *(_cover_column(level) for level in INTERVAL_LEVELS),
        *(_pit_column(i) for i in range(PIT_BINS)),
        "corr_error_trailing",
        "corr_bias_trailing",
        "corr_error_forward",
        "ew_var_exceeded",
    )
    parameters = {
        "trailing_std_window": TRAILING_STD_WINDOW,
        "interval_levels": list(INTERVAL_LEVELS),
        "var_level": VAR_LEVEL,
        "pit_bins": PIT_BINS,
        "energy_subsample": ENERGY_SUBSAMPLE,
    }

    def __call__(
        self, inputs: OriginInputs, cvar_data: CvarData
    ) -> Mapping[str, float]:
        """Score ``cvar_data.R`` (assets x scenarios) at one origin.

        Raises
        ------
        ValueError
            If the look-back window is shorter than ``TRAILING_STD_WINDOW``,
            the forward window holds fewer than two returns, or a correlation
            is undefined because a column is constant.
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
        samples = np.asarray(cvar_data.R).T
        realized = forward[0]
        trailing_std = window[-TRAILING_STD_WINDOW:].std(axis=0)

        scores = {
            "crps": float((sample_crps(samples, realized) / trailing_std).mean()),
            "energy_score": energy_score(
                samples / trailing_std,
                realized / trailing_std,
                subsample=ENERGY_SUBSAMPLE,
                rng=np.random.default_rng(inputs.seed),
            ),
            "spread": float(np.median(samples.std(axis=0) / trailing_std)),
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
        return scores

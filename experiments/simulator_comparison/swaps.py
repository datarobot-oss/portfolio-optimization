# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependence swap arms: one simulator's per-asset distributions in another's rank order.

The swap arms attribute a difference between JointFM and the window to JointFM's
per-asset distributions (its marginals) or to how its assets move together (its
dependence). They cross two marginal donors with two dependence donors, the
window (``no_fit``) and JointFM, in a two-by-two design (``SWAP_DONORS`` in
``arms.py``). A cell takes, per asset, its marginal donor's scenario values and
reorders them into its dependence donor's rank order: the Schaake shuffle
(Clark et al., 2004, "The Schaake shuffle: a method for reconstructing
space-time variability in forecasted precipitation and temperature fields",
J. Hydrometeorology 5(1)). Every asset keeps its marginal donor's distribution
exactly, and the scenarios' joint rank structure, their empirical copula, is
the dependence donor's.

The JointFM corner is the ``jointfm`` arm itself, since its own rank order leaves
it unchanged, so three cells are arms of their own. The window's next-day
scenarios are its own rows, fewer than ``num_scen``, so both donors are first
stretched to ``num_scen`` rows by repeating each row equally often. A stretched
rank template repeats ranks; their ties are broken at random, independently per
asset, which spreads the scenarios of one repeated template row over that row's
band of ranks independently per asset (the checkerboard copula of the template).
The window corner goes through the same construction as the two swaps, so its
difference from ``no_fit`` measures what the construction itself adds.
"""

from collections.abc import Mapping

import numpy as np

from portfolio_optimization.cvar_data import CvarData
from portfolio_optimization.cvar_utils import generate_cvar_data
from portfolio_optimization.settings import ScenarioGenerationSettings

# Separates the tie-breaking draws from every simulator's draws of the same seed.
SWAP_STREAM = 0x53574150


def swap_rng(seed: int, horizon: int) -> np.random.Generator:
    """Tie-breaking generator of a swap at ``horizon`` under scenario seed ``seed``."""
    return np.random.default_rng([seed, SWAP_STREAM, horizon])


def stretch(samples: np.ndarray, num_scen: int) -> np.ndarray:
    """``samples`` with each row repeated equally often to ``num_scen`` rows.

    Row ``i`` of ``n`` appears ``floor`` or ``ceil`` of ``num_scen / n`` times,
    in a contiguous block; ``num_scen == n`` returns the rows unchanged.

    Raises
    ------
    ValueError
        If ``samples`` holds more rows than ``num_scen``, which would drop
        scenarios.
    """
    rows = samples.shape[0]
    if rows > num_scen:
        raise ValueError(
            f"A donor with {rows} scenarios cannot be stretched to {num_scen}."
        )
    return samples[np.arange(num_scen) * rows // num_scen]


def schaake_shuffle(
    marginals: np.ndarray,
    dependence: np.ndarray,
    *,
    num_scen: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """The marginal donor's values per asset in the dependence donor's rank order.

    Both donors are stretched to ``num_scen`` rows. Per asset, the scenario that
    holds the k-th smallest value of the stretched dependence donor receives
    the k-th smallest value of the stretched marginal donor; ties in the
    dependence donor are broken by ``rng``, independently per asset.

    Parameters
    ----------
    marginals, dependence : np.ndarray
        Donor scenarios, shape ``(scenarios, assets)``, the same assets in the
        same column order; the scenario counts may differ.
    num_scen : int
        Scenarios of the result, at least either donor's count.
    rng : np.random.Generator
        Breaks ties of the stretched dependence donor.

    Returns
    -------
    np.ndarray
        Shape ``(num_scen, assets)``. Each column is a permutation of the
        stretched marginal donor's column. When both donors are the same
        array, the result is that array stretched, whatever the ties.

    Raises
    ------
    ValueError
        If a donor is not two-dimensional, the donors' asset counts differ, a
        donor holds a non-finite value, or as ``stretch`` raises.
    """
    if marginals.ndim != 2 or dependence.ndim != 2:
        raise ValueError(
            f"Donors must be (scenarios, assets); got {marginals.shape} and "
            f"{dependence.shape}."
        )
    if marginals.shape[1] != dependence.shape[1]:
        raise ValueError(
            f"The marginal donor holds {marginals.shape[1]} assets and the "
            f"dependence donor {dependence.shape[1]}."
        )
    if not (np.isfinite(marginals).all() and np.isfinite(dependence).all()):
        raise ValueError("A donor holds a non-finite scenario value.")
    values = np.sort(stretch(marginals, num_scen), axis=0)
    template = stretch(dependence, num_scen)
    # lexsort sorts by its last key first, so the random key only breaks ties.
    order = np.lexsort((rng.random(template.shape), template), axis=0)
    shuffled = np.empty_like(values)
    np.put_along_axis(shuffled, order, values, axis=0)
    return shuffled


def swap_scenarios(
    marginals: Mapping[int, np.ndarray],
    dependence: Mapping[int, np.ndarray],
    *,
    num_scen: int,
    seed: int,
) -> dict[int, np.ndarray]:
    """A swap cell's scenarios at every horizon, from its donors' at that horizon.

    Parameters
    ----------
    marginals, dependence : Mapping
        Each donor's scenarios by horizon, as ``horizon_scenarios`` returns them.
    num_scen : int
        Scenarios of the cell at every horizon.
    seed : int
        The origin's scenario seed; horizon ``h`` breaks ties with
        ``swap_rng(seed, h)``.

    Raises
    ------
    ValueError
        If the donors cover different horizons, or as ``schaake_shuffle`` raises.
    """
    if list(marginals) != list(dependence):
        raise ValueError(
            f"The donors cover horizons {list(marginals)} and {list(dependence)}."
        )
    return {
        horizon: schaake_shuffle(
            marginals[horizon],
            dependence[horizon],
            num_scen=num_scen,
            rng=swap_rng(seed, horizon),
        )
        for horizon in marginals
    }


class SwapScenarioGenerator:
    """A swap cell's scenarios for ``rebalance_portfolio``'s ``scenario_generator``.

    Called like ``generate_cvar_data`` with the marginal donor's settings, it
    also generates the dependence donor's next-day scenarios from the same
    window, shuffles the marginal donor's values into their rank order, and
    keeps the marginal donor's expected returns, so the cell's ``CvarData``
    differs from its marginal donor's only in how its assets move together.
    """

    def __init__(self, dependence: ScenarioGenerationSettings, num_scen: int):
        """``dependence`` generates the dependence donor; ``num_scen`` sizes the cell."""
        self.dependence = dependence
        self.num_scen = num_scen

    def __call__(
        self,
        returns_dict: dict,
        scenario_generation_settings: ScenarioGenerationSettings,
        cash_return: float = 0.0,
    ) -> dict:
        """``returns_dict`` with the cell's ``cvar_data`` added, as the library does.

        Ties are broken with ``swap_rng`` of the marginal donor's seed at the
        next day, so every decision of a run with a fixed seed breaks them alike.

        Raises
        ------
        ValueError
            If the marginal donor's settings carry no seed, or as
            ``schaake_shuffle`` or a donor's simulator raises.
        """
        seed = scenario_generation_settings.seed
        if seed is None:
            raise ValueError("A swap cell needs a seeded marginal donor.")
        marginals = generate_cvar_data(
            dict(returns_dict), scenario_generation_settings, cash_return=cash_return
        )["cvar_data"]
        dependence = generate_cvar_data(
            dict(returns_dict), self.dependence, cash_return=cash_return
        )["cvar_data"]
        shuffled = schaake_shuffle(
            np.asarray(marginals.R).T,
            np.asarray(dependence.R).T,
            num_scen=self.num_scen,
            rng=swap_rng(seed, 1),
        )
        returns_dict["cvar_data"] = CvarData(
            mean=marginals.mean,
            R=shuffled.T,
            p=np.ones(self.num_scen) / self.num_scen,
            cash_return=cash_return,
        )
        return returns_dict

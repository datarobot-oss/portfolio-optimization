# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The independence control: the window's scenarios with all dependence removed.

The rebalancing backtest can judge a simulator's dependence only if dependence
moves its readings at all. The positive control answers that: it permutes each
asset's scenario rows independently, which keeps every asset's values, and so
its distribution and its mean, and destroys how the assets move together. A
reading that the control moves beyond its noise against the window (``no_fit``)
can see dependence; one it does not move cannot judge it.

The permutation of every asset is drawn from the scenario seed and a stream of
its own (``INDEPENDENCE_STREAM``), so it is reproducible and independent of
every simulator's draws under the same seed.
"""

from collections.abc import Mapping

import numpy as np

from portfolio_optimization.cvar_data import CvarData
from portfolio_optimization.cvar_utils import generate_cvar_data
from portfolio_optimization.settings import ScenarioGenerationSettings

# Separates the permutations from every simulator's draws of the same seed.
INDEPENDENCE_STREAM = 0x494E4450


def independence_rng(seed: int, horizon: int) -> np.random.Generator:
    """Permutation generator at ``horizon`` under scenario seed ``seed``."""
    return np.random.default_rng([seed, INDEPENDENCE_STREAM, horizon])


def permute_assets(samples: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``samples`` (scenarios x assets) with each asset's rows permuted independently.

    Raises
    ------
    ValueError
        If ``samples`` is not two-dimensional.
    """
    if samples.ndim != 2:
        raise ValueError(
            f"Scenarios must be (scenarios, assets); got shape {samples.shape}."
        )
    return rng.permuted(samples, axis=0)


def independent_scenarios(
    window: Mapping[int, np.ndarray], *, seed: int
) -> dict[int, np.ndarray]:
    """The control's scenarios at every horizon, from the window's at that horizon.

    Horizon ``h`` permutes with ``independence_rng(seed, h)``.
    """
    return {
        horizon: permute_assets(samples, independence_rng(seed, horizon))
        for horizon, samples in window.items()
    }


class IndependentScenarioGenerator:
    """The control's scenarios for ``rebalance_portfolio``'s ``scenario_generator``.

    Called like ``generate_cvar_data`` with the window's (``no_fit``) settings,
    it permutes each asset's rows of the window's scenarios and keeps their
    expected returns, so its ``CvarData`` differs from the window's only in how
    its assets move together.
    """

    def __call__(
        self,
        returns_dict: dict,
        scenario_generation_settings: ScenarioGenerationSettings,
        cash_return: float = 0.0,
    ) -> dict:
        """``returns_dict`` with the control's ``cvar_data`` added, as the library does.

        Every decision of a run with a fixed seed permutes with
        ``independence_rng`` of that seed at the next day.

        Raises
        ------
        ValueError
            If the settings carry no seed.
        """
        seed = scenario_generation_settings.seed
        if seed is None:
            raise ValueError("The independence control needs a seeded window.")
        window = generate_cvar_data(
            dict(returns_dict), scenario_generation_settings, cash_return=cash_return
        )["cvar_data"]
        permuted = permute_assets(np.asarray(window.R).T, independence_rng(seed, 1))
        returns_dict["cvar_data"] = CvarData(
            mean=window.mean, R=permuted.T, p=window.p, cash_return=cash_return
        )
        return returns_dict

# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each arm's scenarios of the simple returns over several horizons after an origin.

A rebalanced portfolio is held for weeks, so the comparison scores every arm at
several horizons, not only the next day. JointFM forecasts every horizon in one
request (``generate_horizon_samples_jointfm``): the forecast log price at h
minus the last observed one is the h-day log return, which the library turns
into a simple return. Every other arm turns its one-day simulator into an h-day
forecast by one rule that needs no new modeling: the h-day scenario compounds h
independent one-day draws of the arm's own simulator, ``prod(1 + r) - 1``.
The next day is the arm's own one-day scenarios, exactly as
``generate_cvar_data`` returns them, so the next-day readings do not depend on
which other horizons a run scores.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from portfolio_optimization.cvar_utils import (
    generate_cvar_data,
    generate_horizon_samples_jointfm,
)

from .arms import Arm, scenario_settings

# Separates the draws of the multi-day sums from the next-day scenarios of the
# same seed.
HORIZON_STREAM = 0x484F525A


def day_seed(seed: int, day: int) -> int:
    """Seed of day ``day`` of the multi-day sums at an origin of scenario seed ``seed``."""
    return int(np.random.SeedSequence([seed, HORIZON_STREAM, day]).generate_state(1)[0])


def _one_day(arm: Arm, returns_dict: dict, *, num_scen: int, seed: int) -> np.ndarray:
    """The arm's one-day scenarios (scenarios x assets) under ``seed``."""
    settings = scenario_settings(arm, num_scen=num_scen, seed=seed)
    # generate_cvar_data adds cvar_data to the dict it gets; arms must not see
    # another arm's result.
    cvar_data = generate_cvar_data(dict(returns_dict), settings)["cvar_data"]
    return np.asarray(cvar_data.R).T


def horizon_scenarios(
    arm: Arm,
    returns_dict: dict,
    *,
    num_scen: int,
    seed: int,
    horizons: Sequence[int],
) -> Mapping[int, np.ndarray]:
    """Scenarios of the simple return over every horizon in ``horizons``.

    Parameters
    ----------
    arm : Arm
        The simulator.
    returns_dict : dict
        The look-back window's simple returns as ``calculate_returns`` builds
        them.
    num_scen : int
        Scenarios per generating arm, and per multi-day horizon of every arm.
    seed : int
        The origin's scenario seed. The next day's scenarios use it as every
        run without other horizons does; day ``d`` of the multi-day sums uses
        ``day_seed(seed, d)``.
    horizons : sequence of int
        Trading days after the origin, starting at 1 and strictly increasing.

    Returns
    -------
    Mapping
        Horizon to scenarios, shape ``(scenarios, assets)``. The next day holds
        the arm's own one-day scenarios: the window's returns for ``no_fit``,
        ``num_scen`` draws otherwise. A multi-day horizon of a generating arm
        holds ``num_scen`` compounded independent one-day draws, one per day,
        read off one running product over the days up to the longest horizon;
        for ``no_fit``, whose one-day scenarios are the window's fixed returns,
        each day's draw resamples them uniformly with replacement.
        For JointFM every horizon comes from one request, whose horizons the
        service draws independently.

    Raises
    ------
    ValueError
        If ``returns_dict`` holds other than simple (``LINEAR``) returns, if
        ``horizons`` does not start at 1 or is not strictly increasing, if a
        one-day draw entering a multi-day horizon is at or below -100 %, a
        non-positive price that cannot compound, or as the arm's simulator
        raises.
    """
    if returns_dict["return_type"] != "LINEAR":
        raise ValueError(
            f"Horizon scenarios compound simple returns; the window holds "
            f"{returns_dict['return_type']} returns."
        )
    horizons = list(horizons)
    if horizons[0] != 1 or any(b <= a for a, b in zip(horizons, horizons[1:])):
        raise ValueError(
            f"horizons {horizons} must start at 1 and be strictly increasing."
        )
    if arm.fit_type == "jointfm":
        settings = scenario_settings(arm, num_scen=num_scen, seed=seed)
        samples = generate_horizon_samples_jointfm(
            num_scen,
            returns_dict["returns"],
            returns_dict["return_type"],
            returns_dict["freq"],
            settings.jointfm_settings,
            horizons,
            seed=seed,
        )
        return {h: samples[:, index, :] for index, h in enumerate(horizons)}

    next_day = _one_day(arm, returns_dict, num_scen=num_scen, seed=seed)
    scenarios = {1: next_day}
    if horizons[-1] == 1:
        return scenarios
    growth = np.ones((num_scen, next_day.shape[1]))
    resample = np.random.default_rng([seed, HORIZON_STREAM])
    for day in range(1, horizons[-1] + 1):
        if arm.fit_type == "no_fit":
            draw = next_day[resample.integers(0, len(next_day), num_scen)]
        else:
            draw = _one_day(
                arm, returns_dict, num_scen=num_scen, seed=day_seed(seed, day)
            )
        if (draw <= -1).any():
            raise ValueError(
                f"{arm.name} drew a one-day return at or below -100 % on day "
                f"{day}, a non-positive price that cannot compound."
            )
        growth *= 1 + draw
        if day > 1 and day in horizons:
            scenarios[day] = growth - 1
    return scenarios

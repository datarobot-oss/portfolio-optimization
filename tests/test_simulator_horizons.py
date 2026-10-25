# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-horizon scenarios of the simulator comparison in experiments/, with
the fake JointFM client of conftest.py."""

import numpy as np
import pandas as pd
import pytest

from experiments.simulator_comparison import horizons as horizons_module
from experiments.simulator_comparison.arms import ARMS_BY_NAME, scenario_settings
from experiments.simulator_comparison.horizons import (
    HORIZON_STREAM,
    day_seed,
    horizon_scenarios,
)
from experiments.simulator_comparison.trials import RETURNS_SETTINGS
from portfolio_optimization.cvar_utils import generate_cvar_data
from portfolio_optimization.utils import calculate_returns

NUM_SCEN = 50
SEED = 9
HORIZONS = (1, 3, 5)
WINDOW = 60
ASSETS = 3


@pytest.fixture()
def returns_dict():
    """A short random-walk window's log returns, as calculate_returns builds them."""
    rng = np.random.default_rng(4)
    log_prices = np.cumsum(rng.normal(0.0, 0.01, size=(WINDOW + 1, ASSETS)), axis=0)
    prices = pd.DataFrame(
        100 * np.exp(log_prices),
        index=pd.bdate_range("2020-01-01", periods=WINDOW + 1),
        columns=[f"A{i}" for i in range(ASSETS)],
    )
    return calculate_returns(prices, returns_compute_settings=RETURNS_SETTINGS)


def _one_day(name, returns_dict, seed, num_scen=NUM_SCEN):
    """The arm's one-day scenarios (scenarios x assets) straight from the library."""
    settings = scenario_settings(ARMS_BY_NAME[name], num_scen=num_scen, seed=seed)
    return generate_cvar_data(dict(returns_dict), settings)["cvar_data"].R.T


@pytest.mark.parametrize("name", ["no_fit-cpu", "gaussian-cpu", "kde-cpu"])
def test_next_day_is_the_arms_own_scenarios(returns_dict, name):
    """The next day is exactly what generate_cvar_data returns under the seed."""
    scenarios = horizon_scenarios(
        ARMS_BY_NAME[name],
        returns_dict,
        num_scen=NUM_SCEN,
        seed=SEED,
        horizons=HORIZONS,
    )
    np.testing.assert_array_equal(scenarios[1], _one_day(name, returns_dict, SEED))
    assert "cvar_data" not in returns_dict


@pytest.mark.parametrize("name", ["gaussian-cpu", "kde-cpu"])
def test_multi_day_sums_independent_daily_draws(returns_dict, name):
    """An h-day scenario sums the first h daily draws, each under its day seed."""
    scenarios = horizon_scenarios(
        ARMS_BY_NAME[name],
        returns_dict,
        num_scen=NUM_SCEN,
        seed=SEED,
        horizons=HORIZONS,
    )
    days = [_one_day(name, returns_dict, day_seed(SEED, d)) for d in range(1, 6)]
    np.testing.assert_allclose(scenarios[3], sum(days[:3]))
    np.testing.assert_allclose(scenarios[5], sum(days))
    assert set(scenarios) == set(HORIZONS)


def test_no_fit_multi_day_resamples_the_window(returns_dict):
    """no_fit's daily draws resample the window's rows uniformly with replacement."""
    scenarios = horizon_scenarios(
        ARMS_BY_NAME["no_fit-cpu"],
        returns_dict,
        num_scen=NUM_SCEN,
        seed=SEED,
        horizons=HORIZONS,
    )
    window = returns_dict["returns"].to_numpy()
    rng = np.random.default_rng([SEED, HORIZON_STREAM])
    rows = [rng.integers(0, WINDOW, NUM_SCEN) for _ in range(5)]
    np.testing.assert_allclose(scenarios[3], sum(window[r] for r in rows[:3]))
    np.testing.assert_allclose(scenarios[5], sum(window[r] for r in rows))
    assert scenarios[1].shape == (WINDOW, ASSETS)
    assert scenarios[5].shape == (NUM_SCEN, ASSETS)


def test_gaussian_covariance_grows_with_the_horizon(returns_dict):
    """Summing h gaussian days scales mean and covariance by h."""
    num_scen = 20_000
    scenarios = horizon_scenarios(
        ARMS_BY_NAME["gaussian-cpu"],
        returns_dict,
        num_scen=num_scen,
        seed=SEED,
        horizons=(1, 8),
    )
    covariance = np.asarray(returns_dict["covariance"])
    np.testing.assert_allclose(
        np.cov(scenarios[8].T), 8 * covariance, atol=0.05 * 8 * covariance.max()
    )
    np.testing.assert_allclose(
        scenarios[8].mean(axis=0),
        8 * np.asarray(returns_dict["mean"]),
        atol=4 * np.sqrt(8 * covariance.max() / num_scen),
    )


def test_jointfm_answers_every_horizon_in_one_request(returns_dict, fake_jointfm):
    """JointFM's h-day scenario is its forecast log price at h minus the last one."""
    fake_jointfm.envelope.n_input = 1000
    scenarios = horizon_scenarios(
        ARMS_BY_NAME["jointfm-cpu"],
        returns_dict,
        num_scen=NUM_SCEN,
        seed=SEED,
        horizons=HORIZONS,
    )
    (call,) = fake_jointfm.forecast_calls
    last = len(call["history"]) - 1
    assert call["query_times"] == [last + h for h in HORIZONS]
    assert call["seed"] == SEED
    log_returns = call["samples"] - call["history"].iloc[-1].to_numpy()
    for index, horizon in enumerate(HORIZONS):
        np.testing.assert_allclose(scenarios[horizon], log_returns[:, index, :])


def test_next_day_alone_draws_nothing_more(returns_dict, monkeypatch):
    """Without multi-day horizons an arm costs one generate_cvar_data call."""
    calls = []

    def counting(*args, **kwargs):
        """generate_cvar_data that records each call."""
        calls.append(args)
        return generate_cvar_data(*args, **kwargs)

    monkeypatch.setattr(horizons_module, "generate_cvar_data", counting)
    horizon_scenarios(
        ARMS_BY_NAME["gaussian-cpu"],
        returns_dict,
        num_scen=NUM_SCEN,
        seed=SEED,
        horizons=(1,),
    )
    assert len(calls) == 1


@pytest.mark.parametrize("horizons", [(3,), (1, 1), (1, 5, 3)])
def test_horizons_must_start_at_the_next_day_and_increase(returns_dict, horizons):
    """Horizons that skip the next day or do not increase are refused."""
    with pytest.raises(ValueError, match="start at 1"):
        horizon_scenarios(
            ARMS_BY_NAME["gaussian-cpu"],
            returns_dict,
            num_scen=NUM_SCEN,
            seed=SEED,
            horizons=horizons,
        )

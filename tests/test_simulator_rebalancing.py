# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the simulator comparison's rebalancing backtest in experiments/,
with CLARABEL and the fake JointFM client of conftest.py."""

import numpy as np
import pandas as pd
import pytest

from experiments.simulator_comparison.config import ExperimentConfig
from experiments.simulator_comparison.rebalancing import (
    METRICS,
    backtest_readings,
    run_rebalancing_experiment,
)
from experiments.simulator_comparison.rebalancing_report import (
    write_rebalancing_report,
)
from experiments.simulator_comparison.table import read_results
from experiments.simulator_comparison.trials import draw_trading_window, draw_trial

ROWS = 300
TRADING_DAYS = 20
LOOK_FORWARD = 5
ARMS = ("no_fit-cpu", "gaussian-cpu", "jointfm")


@pytest.fixture()
def prices_path(tmp_path):
    rng = np.random.default_rng(11)
    dates = pd.date_range("2019-01-01", periods=ROWS, freq="B")
    # Positive drift keeps the risk-aversion calibration defined.
    log_prices = np.cumsum(rng.normal(0.001, 0.01, size=(ROWS, 6)), axis=0)
    frame = pd.DataFrame(
        100.0 * np.exp(log_prices), index=dates, columns=[f"A{i}" for i in range(6)]
    )
    path = tmp_path / "prices.csv"
    frame.to_csv(path)
    return path


def _config(tmp_path, prices_path, **rebalancing):
    values = dict(
        output=tmp_path / "rebalancing.parquet",
        arms=ARMS,
        trials=2,
        trading_days=TRADING_DAYS,
        look_forward_window=LOOK_FORWARD,
        look_back_days=90,
        pct_change_threshold=-0.005,
        transaction_cost=0.001,
        risk_free_rate=0.0,
        solver="CLARABEL",
        cvar=dict(
            w_min=0.0,
            w_max=0.6,
            c_min=0.0,
            c_max=1.0,
            L_tar=1.6,
            T_tar=1.0,
            confidence=0.95,
            risk_aversion=1.0,
        ),
    )
    values.update(rebalancing)
    return ExperimentConfig(
        prices=prices_path,
        output=tmp_path / "results.parquet",
        trials=2,
        origins=2,
        assets=3,
        look_back=64,
        forward=8,
        num_scen=50,
        seed=5,
        arms=("no_fit-cpu",),
        rebalancing=values,
    )


def test_window_is_reproducible_and_fits(prices_path):
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    kwargs = dict(seed=5, trading_days=TRADING_DAYS, look_back_days=90)
    window = draw_trading_window(prices, 1, **kwargs)
    assert window == draw_trading_window(prices, 1, **kwargs)
    assert window.end - window.start == TRADING_DAYS
    assert window.end <= ROWS - 1
    assert prices.index[window.start] - prices.index[0] >= pd.Timedelta(days=90)


def test_runs_every_arm_on_identical_trials(tmp_path, prices_path, fake_jointfm):
    fake_jointfm.envelope.n_input = 1000
    config = _config(tmp_path, prices_path)
    frame = run_rebalancing_experiment(config, command=["first"])

    assert len(frame) == 2 * len(ARMS)
    per_trial = frame.groupby("trial")
    for column in ("assets_hash", "trading_start", "trading_end", "risk_aversion"):
        assert (per_trial[column].nunique() == 1).all(), column
    assert set(frame["arm"]) == {"no_fit-cpu", "gaussian-cpu", "jointfm-cpu"}
    steps = TRADING_DAYS // LOOK_FORWARD
    assert all(len(returns) == steps for returns in frame["step_returns"])
    for row in frame.itertuples():
        compounded = np.prod(1 + np.asarray(row.step_returns)) - 1
        assert compounded == pytest.approx(row.total_return)
        assert row.mean_step_return == pytest.approx(np.mean(row.step_returns))
    assert np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all()
    # Every JointFM request carries the trial's rebalancing seed.
    jointfm_seeds = {call["seed"] for call in fake_jointfm.forecast_calls}
    assert len(jointfm_seeds) == 2

    again = run_rebalancing_experiment(config, command=["again"])
    assert len(again) == len(frame)
    _, metadata = read_results(config.rebalancing.output)
    assert metadata.identity["rebalancing"]["solver"] == "CLARABEL"
    assert "cvxpy" in metadata.identity["packages"]

    files = write_rebalancing_report(config.rebalancing.output, tmp_path / "report")
    names = {path.name for path in files}
    for plot in ("rebalancing_differences", "rebalancing_paths"):
        assert {f"{plot}.png", f"{plot}.svg"} <= names
    comparisons = pd.read_csv(tmp_path / "report" / "rebalancing_comparisons.csv")
    assert set(comparisons["reference"]) == {"no_fit-cpu"}


def test_other_rebalancing_settings_are_refused(tmp_path, prices_path, fake_jointfm):
    fake_jointfm.envelope.n_input = 1000
    run_rebalancing_experiment(_config(tmp_path, prices_path, arms=("no_fit-cpu",)))
    with pytest.raises(ValueError, match="rebalancing"):
        run_rebalancing_experiment(
            _config(tmp_path, prices_path, arms=("no_fit-cpu",), transaction_cost=0.0)
        )


def test_partial_steps_are_refused(tmp_path, prices_path):
    with pytest.raises(ValueError, match="multiple of look_forward_window"):
        _config(tmp_path, prices_path, trading_days=22)


def test_standard_metrics_of_a_known_value_path():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    value = pd.Series([1.0, 1.1, 0.99, 1.089, 1.2], index=dates)
    results = pd.DataFrame(
        {"optimal_portfolio": [None, None], "re_optimized": [False, True]},
        index=dates[[0, 2]],
    )
    readings = backtest_readings(results, value, value, steps=2, risk_free_rate=0.0)
    daily = np.array([0.1, -0.1, 0.1, 1.2 / 1.089 - 1])
    assert readings["step_returns"] == pytest.approx([-0.01, 1.2 / 0.99 - 1])
    assert readings["total_return"] == pytest.approx(0.2)
    assert readings["annualized_return"] == pytest.approx(1.2 ** (252 / 4) - 1)
    assert readings["max_drawdown"] == pytest.approx(0.1)
    assert readings["sharpe_ratio"] == pytest.approx(
        daily.mean() / daily.std(ddof=1) * np.sqrt(252)
    )
    downside = np.sqrt(np.mean(np.minimum(daily, 0) ** 2))
    assert readings["sortino_ratio"] == pytest.approx(
        daily.mean() / downside * np.sqrt(252)
    )
    assert readings["calmar_ratio"] == pytest.approx(
        readings["annualized_return"] / 0.1
    )
    assert readings["rebalances"] == 1


def test_rebalancing_trials_are_independent_of_scenario_trials(
    tmp_path, prices_path, fake_jointfm
):
    fake_jointfm.envelope.n_input = 1000
    config = _config(tmp_path, prices_path, arms=("no_fit-cpu",), trials=3)
    frame = run_rebalancing_experiment(config)
    assert sorted(frame["trial"]) == [0, 1, 2]
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    scenario_trial = draw_trial(
        prices,
        0,
        seed=config.seed,
        n_assets=config.assets,
        n_origins=config.origins,
        look_back=config.look_back,
        forward=config.forward,
    )
    first = frame.loc[frame["trial"] == 0, "assets_hash"].item()
    assert first == scenario_trial.assets_hash

    extended = run_rebalancing_experiment(
        _config(tmp_path, prices_path, arms=("no_fit-cpu",), trials=4)
    )
    assert sorted(extended["trial"]) == [0, 1, 2, 3]


def test_service_switch_during_a_backtest_is_refused(
    tmp_path, prices_path, fake_jointfm
):
    fake_jointfm.envelope.n_input = 1000
    forecast = fake_jointfm.forecast

    def forecast_then_switch(*args, **kwargs):
        fake_jointfm.device = "cuda:0"
        return forecast(*args, **kwargs)

    fake_jointfm.forecast = forecast_then_switch
    config = _config(tmp_path, prices_path, arms=("jointfm",))
    with pytest.raises(ValueError, match="changed during the jointfm-cpu"):
        run_rebalancing_experiment(config)
    assert not config.rebalancing.output.exists()

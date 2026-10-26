# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the chosen-portfolio record and readings of the rebalancing backtest,
against a hand-computed scenario set and return path."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from experiments.simulator_comparison.chosen import (
    DECISION_COLUMNS,
    Decision,
    DecisionRecorder,
    chosen_readings,
    effective_holdings,
    scenario_risk,
)
from experiments.simulator_comparison.rebalancing_report import chosen_risk
from portfolio_optimization.cvar_data import CvarData
from portfolio_optimization.portfolio import Portfolio

TICKERS = ("A", "B")
# 21 scenarios put the 5 % quantile exactly on the second-worst scenario.
SCENARIOS = 21


def _portfolio_return(returns, weights):
    """The definition: the weighted simple returns, cash earning nothing."""
    return np.asarray(returns) @ np.asarray(weights)


def _scenarios():
    """Asset A's simple returns step by 1 % from -10 %; asset B never moves."""
    simple = (np.arange(SCENARIOS) - 10) / 100
    return np.column_stack([simple, np.zeros(SCENARIOS)])


def test_effective_holdings_counts_equal_positions_whatever_cash_and_leverage():
    assert effective_holdings(np.array([0.3, 0.3, 0.3, 0.0])) == pytest.approx(3)
    assert effective_holdings(np.array([0.4, 0.4, -0.4, -0.4])) == pytest.approx(4)
    assert effective_holdings(np.array([0.6, -0.2])) == pytest.approx(0.64 / 0.40)
    with pytest.raises(ValueError, match="no asset"):
        effective_holdings(np.zeros(2))


def test_scenario_risk_of_a_hand_computed_scenario_set():
    risk = scenario_risk(_scenarios(), np.array([0.5, 0.3]), confidence=0.95)
    # Portfolio simple return 0.5 * (j - 10) / 100 in scenario j.
    returns = 0.005 * (np.arange(SCENARIOS) - 10)
    var = 0.045
    assert risk["var"] == pytest.approx(var)
    # Only the worst scenario lies beyond the VaR; it weighs 1/21 of the 5 % tail.
    assert risk["cvar"] == pytest.approx(var + (0.05 - 0.045) / 21 / 0.05)
    assert risk["mean"] == pytest.approx(returns.mean())
    assert risk["std"] == pytest.approx(returns.std())


def test_recorder_reads_the_optimizer_scenarios_and_the_chosen_weights():
    recorder = DecisionRecorder(confidence=0.95)
    scenarios = _scenarios()
    data = CvarData(
        mean=scenarios.mean(axis=0),
        R=scenarios.T,
        p=np.full(SCENARIOS, 1 / SCENARIOS),
    )
    chosen = Portfolio(tickers=list(TICKERS), weights=np.array([0.5, 0.3]), cash=0.2)
    recorder(pd.Timestamp("2024-01-01"), data, chosen)

    (decision,) = recorder.decisions
    assert decision.date == pd.Timestamp("2024-01-01")
    assert decision.tickers == TICKERS
    assert decision.var == pytest.approx(0.045)
    assert decision.holdings == pytest.approx(0.64 / 0.34)


def test_recorder_refuses_unequal_scenario_probabilities():
    scenarios = _scenarios()
    p = np.full(SCENARIOS, 0.5 / (SCENARIOS - 1))
    p[0] = 0.5
    data = CvarData(mean=scenarios.mean(axis=0), R=scenarios.T, p=p)
    chosen = Portfolio(tickers=list(TICKERS), weights=np.array([0.5, 0.3]), cash=0.2)
    with pytest.raises(ValueError, match="equally likely"):
        DecisionRecorder(confidence=0.95)(pd.Timestamp("2024-01-01"), data, chosen)


def _decision(date, weights, var, mean, std):
    weights = np.asarray(weights, dtype=float)
    return Decision(
        date=pd.Timestamp(date),
        tickers=TICKERS,
        weights=weights,
        var=var,
        cvar=var + 0.01,
        mean=mean,
        std=std,
        holdings=effective_holdings(weights),
    )


def _return_path():
    """Five days after the first decision; the second decision falls on the third."""
    dates = pd.bdate_range("2024-01-02", periods=5)
    return pd.DataFrame(
        {"A": [-0.04, 0.02, -0.01, 0.0, 0.0], "B": [0.0, 0.0, 0.0, -0.05, 0.01]},
        index=dates,
    )


def test_readings_of_a_hand_computed_return_path():
    realized = _return_path()
    first = _decision("2024-01-01", [0.5, 0.5], var=0.01, mean=0.0, std=0.01)
    second = _decision(realized.index[2], [0.0, 0.8], var=0.02, mean=0.001, std=0.02)
    readings = chosen_readings([first, second], realized)

    held_first = _portfolio_return(realized.iloc[:3], first.weights)
    held_second = _portfolio_return(realized.iloc[3:], second.weights)
    # Days 1 (-2 %) and 4 (-4 %) fall below their decisions' VaR.
    assert list(held_first < -0.01) == [True, False, False]
    assert list(held_second < -0.02) == [True, False]
    assert readings["chosen_days"] == [3, 2]
    assert readings["chosen_breaches"] == [1, 1]
    z2_first = float(np.sum((held_first / 0.01) ** 2))
    z2_second = float(np.sum(((held_second - 0.001) / 0.02) ** 2))
    assert readings["chosen_z2"] == pytest.approx([z2_first, z2_second])
    assert readings["chosen_breach_rate"] == pytest.approx(2 / 5)
    assert readings["chosen_volatility_ratio"] == pytest.approx(
        np.sqrt((z2_first + z2_second) / 5)
    )
    # Two equal positions for three days, one for two.
    assert readings["chosen_effective_holdings"] == pytest.approx((2 * 3 + 1 * 2) / 5)
    assert readings["chosen_dates"] == ["2024-01-01", "2024-01-04"]
    assert readings["chosen_var"] == [0.01, 0.02]
    assert readings["chosen_cvar"] == pytest.approx([0.02, 0.03])
    assert readings["chosen_mean"] == [0.0, 0.001]
    assert readings["chosen_std"] == [0.01, 0.02]
    assert readings["chosen_holdings"] == pytest.approx([2, 1])
    assert set(DECISION_COLUMNS) <= set(readings)


def test_a_decision_that_holds_no_day_is_refused():
    realized = _return_path()
    last = realized.index[-1]
    decisions = [
        _decision("2024-01-01", [0.5, 0.5], var=0.01, mean=0.0, std=0.01),
        _decision(last, [0.0, 0.8], var=0.02, mean=0.0, std=0.02),
    ]
    with pytest.raises(ValueError, match="holds no trading day"):
        chosen_readings(decisions, realized)


def test_returns_before_the_first_decision_are_refused():
    realized = _return_path()
    decisions = [_decision(realized.index[1], [0.5, 0.5], var=0.01, mean=0.0, std=0.01)]
    with pytest.raises(ValueError, match="not after the first decision"):
        chosen_readings(decisions, realized)


def test_other_tickers_are_refused():
    realized = _return_path().rename(columns={"B": "C"})
    decisions = [_decision("2024-01-01", [0.5, 0.5], var=0.01, mean=0.0, std=0.01)]
    with pytest.raises(ValueError, match="returns cover"):
        chosen_readings(decisions, realized)


def _runs(days_per_decision):
    """Two arms over two trials with known breach rates and volatility ratios."""
    return pd.DataFrame(
        {
            "arm": ["no_fit-cpu", "no_fit-cpu", "jointfm-cpu", "jointfm-cpu"],
            "trial": [0, 1, 0, 1],
            "chosen_days": days_per_decision,
            "chosen_breach_rate": [0.05, 0.07, 0.10, 0.12],
            "chosen_volatility_ratio": [1.0, 1.2, 1.3, 1.5],
            "chosen_effective_holdings": [10.0, 12.0, 20.0, 22.0],
        }
    )


def test_chosen_risk_pools_runs_and_bands_the_breach_rate():
    table = chosen_risk(_runs([[50, 50]] * 4), confidence=0.95).set_index("arm")

    jointfm = table.loc["jointfm-cpu"]
    assert jointfm["days"] == 200
    assert jointfm["breach_rate"] == pytest.approx(0.11)
    assert jointfm["volatility_ratio"] == pytest.approx(np.sqrt((1.3**2 + 1.5**2) / 2))
    assert jointfm["effective_holdings"] == pytest.approx(21.0)
    low, high = binom.interval(0.95, 200, 0.05)
    assert jointfm["breach_band_low"] == pytest.approx(low / 200)
    assert jointfm["breach_band_high"] == pytest.approx(high / 200)
    assert jointfm["breach_ci_low"] <= 0.11 <= jointfm["breach_ci_high"]
    assert table.loc["no_fit-cpu", "breach_rate"] == pytest.approx(0.06)


def test_chosen_risk_refuses_runs_of_different_lengths():
    with pytest.raises(ValueError, match="one count"):
        chosen_risk(_runs([[50, 50], [50, 40], [50, 50], [50, 50]]), confidence=0.95)


def test_chosen_risk_refuses_runs_without_a_record():
    runs = _runs([[50, 50]] * 4)
    runs.loc[0, "chosen_breach_rate"] = np.nan
    with pytest.raises(ValueError, match="no chosen-portfolio record"):
        chosen_risk(runs, confidence=0.95)

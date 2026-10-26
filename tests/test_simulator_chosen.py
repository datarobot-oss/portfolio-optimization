# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the chosen-portfolio record and readings of the rebalancing backtest,
against a hand-computed scenario set and return path, and of their figure over
the scenario comparison's size curves, for a config and its risk-averse twin."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from experiments.simulator_comparison.analysis import aggregate
from experiments.simulator_comparison.chosen import (
    DECISION_COLUMNS,
    Decision,
    DecisionRecorder,
    chosen_readings,
    effective_holdings,
    scenario_risk,
)
from experiments.simulator_comparison.chosen_report import (
    CHOSEN_BAND_HALF_WIDTH,
    RANDOM_KINDS,
    SPREAD_READING,
    backtest_universes,
    band_gid,
    chosen_points,
    chosen_risk_figure,
    curve_styles,
    next_day_trials,
    point_gid,
)
from experiments.simulator_comparison.config import ExperimentConfig
from experiments.simulator_comparison.plotting import ArmStyle, arm_styles
from experiments.simulator_comparison.portfolio_analysis import size_curves
from experiments.simulator_comparison.portfolio_report import (
    KIND_LABELS,
    READING_LABELS,
)
from experiments.simulator_comparison.rebalancing import run_rebalancing_experiment
from experiments.simulator_comparison.rebalancing_report import (
    chosen_risk,
    complete_runs,
    per_trial_frame,
    write_rebalancing_report,
)
from experiments.simulator_comparison.runner import run_experiment
from experiments.simulator_comparison.scores import PortfolioScores, ScenarioScores
from experiments.simulator_comparison.table import read_results
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


def test_chosen_points_put_the_spread_on_the_size_curves_scale():
    chosen = pd.DataFrame(
        {
            "arm": ["no_fit-cpu"],
            "var_level": [1 - 0.95],
            "breach_rate": [0.08],
            "breach_ci_low": [0.06],
            "breach_ci_high": [0.10],
            "breach_band_low": [0.04],
            "breach_band_high": [0.06],
            "volatility_ratio": [1.25],
            "volatility_ratio_ci_low": [1.0],
            "volatility_ratio_ci_high": [2.0],
            "effective_holdings": [7.5],
        }
    )
    points = chosen_points(chosen).set_index("reading")
    assert (points["holdings"] == 7.5).all()
    breach = points.loc["breach_05"]
    assert list(breach[["estimate", "ci_low", "ci_high"]]) == [0.08, 0.06, 0.10]
    assert list(breach[["band_low", "band_high"]]) == [0.04, 0.06]
    # Scenario over realized: the reciprocal of realized over scenario volatility.
    spread = points.loc[SPREAD_READING]
    assert list(spread[["estimate", "ci_low", "ci_high"]]) == [0.8, 0.5, 1.0]

    with pytest.raises(ValueError, match="VaR levels"):
        chosen_points(chosen.assign(var_level=0.02))


def test_curves_follow_the_drawn_simulators_whatever_their_device():
    scenario = [
        ArmStyle("no_fit-cpu", "no_fit", "CPU"),
        ArmStyle("gaussian-cpu", "gaussian", "CPU"),
        ArmStyle("jointfm-gpu", "jointfm", "GPU"),
    ]
    drawn = [
        ArmStyle("no_fit-cpu", "no_fit", "CPU"),
        ArmStyle("jointfm-cpu", "jointfm", "CPU"),
    ]
    names = [style.name for style in curve_styles(scenario, drawn)]
    assert names == ["no_fit-cpu", "jointfm-gpu"]


# Long-short portfolios need two longs besides their shorts under these bounds,
# so sizes 3 and 4 hold every kind.
FIGURE_ASSETS = 4
FIGURE_ARMS = ("no_fit-cpu", "gaussian-cpu")


def _figure_prices(directory):
    """Correlated random-walk prices with positive drift, and their cash rates."""
    rows = 300
    rng = np.random.default_rng(12)
    dates = pd.date_range("2019-01-01", periods=rows, freq="B")
    factor = rng.normal(size=(rows, 1))
    noise = rng.normal(size=(rows, FIGURE_ASSETS + 2))
    returns = 0.001 + 0.01 * (np.sqrt(0.5) * factor + np.sqrt(0.5) * noise)
    frame = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=[f"A{i}" for i in range(FIGURE_ASSETS + 2)],
    )
    path = directory / "prices.csv"
    frame.to_csv(path)
    rate_dates = pd.date_range("2018-12-03", "2020-03-31", freq="B")
    pd.Series(0.02, index=rate_dates, name="rate").rename_axis("Date").to_csv(
        directory / "rates.csv"
    )
    return path


def _figure_config(directory, prices_path, name, scenario_results, risk_aversion):
    return ExperimentConfig(
        prices=prices_path,
        output=directory / f"{name}.parquet",
        trials=2,
        origins=3,
        assets=FIGURE_ASSETS,
        look_back=64,
        origin_history=64,
        forward=8,
        num_scen=100,
        seed=5,
        arms=FIGURE_ARMS,
        portfolio_sizes=(1, 2, 3, FIGURE_ASSETS),
        horizons=(1, 3),
        rebalancing=dict(
            output=directory / f"{name}_rebalancing.parquet",
            scenario_results=scenario_results,
            arms=FIGURE_ARMS,
            trials=2,
            trading_days=10,
            look_forward_window=5,
            look_back_days=90,
            trigger="every_step",
            pct_change_threshold=-0.005,
            transaction_cost=0.001,
            risk_free_rates=directory / "rates.csv",
            solver="CLARABEL",
            cvar=dict(
                w_min=-0.3,
                w_max=0.6,
                c_min=0.0,
                c_max=0.4,
                L_tar=1.6,
                T_tar=1.0,
                confidence=0.95,
                risk_aversion=risk_aversion,
            ),
        ),
    )


@pytest.fixture(scope="module")
def configs(tmp_path_factory):
    """A config with its scenario comparison and backtest, and its risk-averse twin.

    The twin runs only its backtest, so its report reads the base's scenario
    results, as the shipped risk-averse twins do.
    """
    directory = tmp_path_factory.mktemp("chosen_figure")
    prices_path = _figure_prices(directory)
    base_output = directory / "base.parquet"
    base = _figure_config(directory, prices_path, "base", base_output, 1.0)
    twin = _figure_config(directory, prices_path, "twin", base_output, 100.0)
    portfolios = PortfolioScores(
        base.portfolio_sizes, base.rebalancing.cvar, base.assets
    )
    run_experiment(base, [ScenarioScores(), portfolios])
    for config in (base, twin):
        run_rebalancing_experiment(config)
    return {"base": base, "twin": twin}


def _panel(figure, reading, kind):
    """The axes whose title names ``reading`` and ``kind``."""
    title = f"{READING_LABELS[reading][0]} · {KIND_LABELS[kind]}"
    (ax,) = [ax for ax in figure.axes if ax.get_title(loc="left") == title]
    return ax


def _artist(ax, gid):
    (artist,) = [item for item in ax.collections if item.get_gid() == gid]
    return artist


@pytest.mark.parametrize("name", ["base", "twin"])
def test_each_arm_sits_at_its_effective_holdings_over_the_size_curves(
    configs, tmp_path, name
):
    config = configs[name]
    base = configs["base"]
    report = tmp_path / "report"
    files = write_rebalancing_report(
        config.rebalancing.output, config.rebalancing.scenario_results, report
    )
    names = {path.name for path in files}
    assert {"rebalancing_chosen_risk.png", "rebalancing_chosen_risk.svg"} <= names
    chosen = pd.read_csv(report / "rebalancing_chosen_risk.csv")
    curves = pd.read_csv(report / "rebalancing_size_curves.csv")

    # The curves are the next-day size curves of the base's scenario comparison
    # over the backtested trials, which here are all of its trials.
    scenario, _ = read_results(base.output)
    per_trial = aggregate(scenario[scenario["horizon"] == 1])
    expected = size_curves(per_trial)
    expected = expected[expected["kind"].isin(RANDOM_KINDS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(curves, expected, check_dtype=False)
    assert set(curves["kind"]) == set(RANDOM_KINDS)

    rebalancing, metadata = read_results(config.rebalancing.output)
    frame = complete_runs(rebalancing, tuple(metadata.identity["arms"]))
    styles = arm_styles(per_trial_frame(frame))
    figure = chosen_risk_figure(
        curves, chosen_points(chosen), arm_styles(per_trial), styles
    )
    svg = (report / "rebalancing_chosen_risk.svg").read_text()
    points = chosen_points(chosen).set_index(["reading", "arm"])
    for reading in ("breach_05", SPREAD_READING):
        for kind in RANDOM_KINDS:
            ax = _panel(figure, reading, kind)
            drawn = {tuple(np.round(line.get_ydata(), 12)) for line in ax.lines}
            cell = curves[(curves["kind"] == kind) & (curves["reading"] == reading)]
            for arm, rows in cell.groupby("arm"):
                estimates = rows.sort_values("size")["estimate"].to_numpy()
                assert tuple(np.round(estimates, 12)) in drawn, (reading, kind, arm)
            for arm in FIGURE_ARMS:
                point = points.loc[(reading, arm)]
                offsets = _artist(ax, point_gid(arm)).get_offsets()
                np.testing.assert_allclose(
                    offsets, [[point["holdings"], point["estimate"]]]
                )
                low, high = ax.get_xlim()
                assert low < point["holdings"] < high
                if reading == SPREAD_READING:
                    continue
                vertices = _artist(ax, band_gid(arm)).get_paths()[0].vertices
                np.testing.assert_allclose(
                    [vertices[:, 0].min(), vertices[:, 0].max()],
                    [
                        point["holdings"] / CHOSEN_BAND_HALF_WIDTH,
                        point["holdings"] * CHOSEN_BAND_HALF_WIDTH,
                    ],
                )
                np.testing.assert_allclose(
                    [vertices[:, 1].min(), vertices[:, 1].max()],
                    [point["band_low"], point["band_high"]],
                )
    for arm in FIGURE_ARMS:
        assert f'id="{point_gid(arm)}"' in svg
        assert f'id="{band_gid(arm)}"' in svg
    expected_holdings = frame.groupby("arm")["chosen_effective_holdings"].mean()
    np.testing.assert_allclose(
        chosen.set_index("arm")["effective_holdings"],
        expected_holdings.reindex(chosen["arm"]),
    )


def test_twin_reads_the_base_curves_with_its_own_chosen_portfolios(configs, tmp_path):
    # The twin runs no scenario comparison of its own.
    assert not configs["twin"].output.exists()
    tables = {}
    for name, config in configs.items():
        write_rebalancing_report(
            config.rebalancing.output,
            config.rebalancing.scenario_results,
            tmp_path / name,
        )
        tables[name] = {
            table: pd.read_csv(tmp_path / name / f"rebalancing_{table}.csv")
            for table in ("size_curves", "chosen_risk")
        }
    pd.testing.assert_frame_equal(
        tables["twin"]["size_curves"], tables["base"]["size_curves"]
    )
    assert not tables["twin"]["chosen_risk"].equals(tables["base"]["chosen_risk"])


@pytest.fixture()
def backtest(configs):
    """The base backtest's identity and universes."""
    frame, metadata = read_results(configs["base"].rebalancing.output)
    return metadata.identity, backtest_universes(frame)


def test_scenario_results_that_do_not_exist_are_refused(tmp_path, backtest):
    identity, universes = backtest
    with pytest.raises(ValueError, match="does not exist"):
        next_day_trials(tmp_path / "absent.parquet", identity, universes)


def test_scenario_results_lacking_a_backtested_trial_are_refused(configs, backtest):
    identity, universes = backtest
    universes = pd.concat([universes, pd.Series({2: "0" * 16})])
    with pytest.raises(ValueError, match=r"backtested trials \[2\]"):
        next_day_trials(configs["base"].output, identity, universes)


def test_scenario_results_of_other_universes_are_refused(configs, backtest):
    identity, universes = backtest
    universes = universes.copy()
    universes.iloc[0] = "0" * 16
    with pytest.raises(ValueError, match="other universes"):
        next_day_trials(configs["base"].output, identity, universes)


def test_scenario_results_of_other_prices_are_refused(configs, backtest):
    identity, universes = backtest
    other = {**identity, "prices": {**identity["prices"], "sha256": "0" * 64}}
    with pytest.raises(ValueError, match="another price file"):
        next_day_trials(configs["base"].output, other, universes)


def test_scenario_results_of_other_bounds_are_refused(configs, backtest):
    identity, universes = backtest
    cvar = {**identity["rebalancing"]["cvar"], "w_min": 0.0}
    other = {**identity, "rebalancing": {**identity["rebalancing"], "cvar": cvar}}
    with pytest.raises(ValueError, match="random portfolios within"):
        next_day_trials(configs["base"].output, other, universes)

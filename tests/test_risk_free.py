# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the risk-free rate that cash earns: the rate helpers, the
backtester, and both CVaR solve paths."""

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from portfolio_optimization import risk_free
from portfolio_optimization.backtest import portfolio_backtester
from portfolio_optimization.cvar_optimizer import CVaR
from portfolio_optimization.cvar_parameters import CvarParameters
from portfolio_optimization.cvar_utils import generate_cvar_data
from portfolio_optimization.portfolio import Portfolio
from portfolio_optimization.settings import (
    ApiSettings,
    ReturnsComputeSettings,
    ScenarioGenerationSettings,
)
from portfolio_optimization.utils import calculate_returns

TICKERS = ["A", "B", "C"]
ANNUAL = 0.05
LOG = ReturnsComputeSettings(return_type="LOG", freq=1)
# Above every asset's mean daily return, so a riskless cash position wins.
HIGH_CASH_RETURN = 0.01


@pytest.fixture()
def rates():
    return pd.Series(
        [0.01, 0.05], index=pd.to_datetime(["2024-01-02", "2024-01-03"]), name="rate"
    )


@pytest.fixture()
def gaining_returns_dict():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    log_returns = rng.normal(0.002, 0.005, (len(dates), len(TICKERS)))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(log_returns, axis=0)), index=dates, columns=TICKERS
    )
    returns_dict = calculate_returns(prices, None, LOG)
    assert (returns_dict["mean"] > 0).all()
    return returns_dict


class TestCashReturns:
    def test_period_return_compounds_the_annual_rate(self):
        assert risk_free.period_returns(ANNUAL, "LINEAR") == pytest.approx(
            1.05 ** (1 / 252) - 1
        )
        assert risk_free.period_returns(ANNUAL, "LOG", freq=5) == pytest.approx(
            np.log(1.05) * 5 / 252
        )

    def test_other_return_types_are_refused(self):
        with pytest.raises(NotImplementedError, match="LOG or LINEAR"):
            risk_free.period_returns(ANNUAL, "ABSOLUTE")

    def test_period_earns_the_rate_quoted_before_its_end(self, rates):
        dates = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-08"])
        np.testing.assert_allclose(
            risk_free.cash_returns(rates, dates, "LINEAR"),
            risk_free.period_returns(np.array([0.01, 0.05, 0.05]), "LINEAR"),
        )

    def test_scalar_rate_applies_to_every_period(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        np.testing.assert_allclose(
            risk_free.cash_returns(ANNUAL, dates, "LOG"),
            np.full(3, np.log(1.05) / 252),
        )

    @pytest.mark.parametrize("end", ["2024-01-02", "2024-01-20"])
    def test_periods_outside_the_quotes_are_refused(self, rates, end):
        with pytest.raises(ValueError, match="do not cover"):
            risk_free.cash_returns(rates, pd.to_datetime([end]), "LOG")


class TestRatesFile:
    def test_download_converts_percent_and_drops_holidays(self, tmp_path, monkeypatch):
        fred = tmp_path / "fred.csv"
        fred.write_text(
            "observation_date,DTB3\n2024-01-02,5.25\n2024-01-03,\n2024-01-04,.\n"
            "2024-01-05,5.20\n"
        )
        monkeypatch.setattr(risk_free, "FRED_DTB3_URL", str(fred))
        rates = risk_free.download_risk_free_rates(tmp_path / "rates" / "dtb3.csv")
        assert list(rates.index) == list(pd.to_datetime(["2024-01-02", "2024-01-05"]))
        np.testing.assert_allclose(rates.to_numpy(), [0.0525, 0.052])

    def test_unordered_dates_are_refused(self, tmp_path, rates):
        path = tmp_path / "rates.csv"
        rates.iloc[::-1].rename_axis("Date").to_csv(path)
        with pytest.raises(ValueError, match="strictly increasing"):
            risk_free.load_risk_free_rates(path)

    def test_missing_rates_are_refused(self, tmp_path):
        path = tmp_path / "rates.csv"
        path.write_text("Date,rate\n2024-01-02,0.05\n2024-01-03,\n")
        with pytest.raises(ValueError, match="missing or non-finite"):
            risk_free.load_risk_free_rates(path)


class TestBacktesterCash:
    def test_cash_accrues_dated_rates(self, gaining_returns_dict):
        dates = gaining_returns_dict["returns"].index
        quote_dates = pd.date_range(
            dates[0] - pd.Timedelta(days=7), dates[-1], freq="B"
        )
        rates = pd.Series(
            np.linspace(0.01, 0.05, len(quote_dates)), index=quote_dates, name="rate"
        )
        weights = np.array([0.2, 0.2, 0.1])
        portfolio = Portfolio(name="half", tickers=TICKERS, weights=weights, cash=0.5)
        backtester = portfolio_backtester(
            portfolio,
            gaining_returns_dict,
            risk_free_rate=rates,
            benchmark_portfolios=[],
        )
        result = backtester.backtest_single_portfolio(portfolio)
        expected = gaining_returns_dict[
            "returns"
        ].to_numpy() @ weights + 0.5 * risk_free.cash_returns(rates, dates, "LOG")
        np.testing.assert_allclose(result["returns"].iloc[0], expected)
        # Cash already sits in the returns, so the mean must not add it again.
        assert result["mean portfolio return"].iloc[0] == pytest.approx(expected.mean())

    # An all-cash portfolio has no excess-return spread, so its Sharpe ratio is 0/0.
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    def test_scalar_rate_is_annual(self, gaining_returns_dict):
        portfolio = Portfolio(
            name="cash", tickers=TICKERS, weights=np.zeros(len(TICKERS)), cash=1.0
        )
        backtester = portfolio_backtester(
            portfolio,
            gaining_returns_dict,
            risk_free_rate=ANNUAL,
            benchmark_portfolios=[],
        )
        returns = backtester.backtest_single_portfolio(portfolio)["returns"].iloc[0]
        np.testing.assert_allclose(returns, np.log(1.05) / 252)

    def test_dated_rates_refuse_simulated_scenarios(self, gaining_returns_dict, rates):
        portfolio = Portfolio(
            name="p", tickers=TICKERS, weights=np.full(3, 1 / 3), cash=0.0
        )
        with pytest.raises(ValueError, match="historical"):
            portfolio_backtester(
                portfolio,
                gaining_returns_dict,
                risk_free_rate=rates,
                test_method="gaussian_simulation",
                benchmark_portfolios=[],
                seed=0,
            )


def _solve(returns_dict, cash_return, api_settings, solver_settings=None):
    data = generate_cvar_data(
        dict(returns_dict),
        ScenarioGenerationSettings(fit_type="no_fit"),
        cash_return=cash_return,
    )
    params = CvarParameters(
        w_min=0.0,
        w_max=0.5,
        c_min=0.0,
        c_max=1.0,
        risk_aversion=0.01,
        confidence=0.95,
        L_tar=1.6,
    )
    optimizer = CVaR(returns_dict=data, cvar_params=params, api_settings=api_settings)
    result, portfolio = optimizer.solve_optimization_problem(
        solver_settings, print_results=False
    )
    weights = np.asarray(portfolio.weights, dtype=float).flatten()
    cash = float(np.asarray(portfolio.cash).squeeze())
    return result, weights, cash


@pytest.mark.parametrize(
    ("cash_return", "expected_cash"), [(0.0, 0.0), (HIGH_CASH_RETURN, 1.0)]
)
def test_cvxpy_credits_cash_with_its_return(
    gaining_returns_dict, cash_return, expected_cash
):
    result, weights, cash = _solve(
        gaining_returns_dict,
        cash_return,
        ApiSettings(scale_risk_aversion=False),
        {"solver": cp.CLARABEL, "verbose": False},
    )
    assert cash == pytest.approx(expected_cash, abs=1e-4)
    assert float(result["return"]) == pytest.approx(
        gaining_returns_dict["mean"] @ weights + cash_return * cash, abs=1e-6
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("cash_return", "expected_cash"), [(0.0, 0.0), (HIGH_CASH_RETURN, 1.0)]
)
def test_cuopt_python_credits_cash_with_its_return(
    gaining_returns_dict, cash_return, expected_cash
):
    result, weights, cash = _solve(
        gaining_returns_dict,
        cash_return,
        ApiSettings(api="cuopt_python", scale_risk_aversion=False),
    )
    assert cash == pytest.approx(expected_cash, abs=1e-4)
    assert float(result["return"]) == pytest.approx(
        gaining_returns_dict["mean"] @ weights + cash_return * cash, abs=1e-6
    )

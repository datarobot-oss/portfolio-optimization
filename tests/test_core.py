# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import cvxpy as cp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from portfolio_optimization import mean_variance_optimizer, rebalance, utils
from portfolio_optimization.backtest import portfolio_backtester
from portfolio_optimization.cvar_data import CvarData
from portfolio_optimization.cvar_optimizer import CVaR
from portfolio_optimization.cvar_parameters import CvarParameters
from portfolio_optimization.cvar_utils import (
    JOINTFM_INSTALL_HINT,
    _annualized_sharpe_ratio,
    _import_jointfm_client,
    _select_efficient_frontier_key_portfolios,
    compute_CVaR,
    create_efficient_frontier,
    expected_returns,
    generate_cvar_data,
    generate_horizon_samples_jointfm,
    generate_samples_jointfm,
    normalize_portfolio_weights_to_one,
)
from portfolio_optimization.mean_variance_parameters import MeanVarianceParameters
from portfolio_optimization.portfolio import Portfolio
from portfolio_optimization.settings import (
    ApiSettings,
    JointFMSettings,
    KDESettings,
    ReturnsComputeSettings,
    ScenarioGenerationSettings,
)
from portfolio_optimization.utils import (
    calculate_log_returns,
    calculate_returns,
    compare_results,
    compute_absolute_returns,
)

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Fixtures: small synthetic data shared across tests
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "GOOGL", "MSFT"]


@pytest.fixture()
def price_data():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    base = np.array([150.0, 100.0, 250.0])
    noise = np.random.randn(60, 3) * 0.5
    prices = base + np.cumsum(noise, axis=0)
    return pd.DataFrame(prices, index=dates, columns=TICKERS)


@pytest.fixture()
def returns_dict(price_data):
    settings = ReturnsComputeSettings(return_type="LOG", freq=1)
    return calculate_returns(
        price_data, regime_dict=None, returns_compute_settings=settings
    )


def _losing_price_frame(dates):
    """Prices whose every asset loses on average, so no single-asset return/risk
    ratio is positive."""
    rng = np.random.default_rng(3)
    log_returns = rng.normal(-0.01, 0.005, (len(dates), len(TICKERS)))
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=TICKERS)


@pytest.fixture()
def losing_returns_dict():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    settings = ReturnsComputeSettings(return_type="LOG", freq=1)
    returns_dict = calculate_returns(
        _losing_price_frame(dates), regime_dict=None, returns_compute_settings=settings
    )
    assert (returns_dict["mean"] < 0).all()
    return generate_cvar_data(
        returns_dict, ScenarioGenerationSettings(num_scen=59, fit_type="no_fit")
    )


@pytest.fixture()
def cvar_data(returns_dict):
    np.random.seed(0)
    settings = ScenarioGenerationSettings(num_scen=200, fit_type="gaussian")
    rd = generate_cvar_data(returns_dict, settings)
    return rd["cvar_data"]


@pytest.fixture()
def cvar_params():
    return CvarParameters(
        w_min=0.0,
        w_max=0.5,
        c_min=0.0,
        c_max=1.0,
        risk_aversion=1.0,
        confidence=0.95,
        L_tar=1.6,
    )


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


class TestReturns:
    def test_log_returns_shape(self, price_data):
        ret = calculate_log_returns(price_data, freq=1)
        assert ret.shape == (59, 3)
        assert list(ret.columns) == TICKERS

    def test_log_returns_values(self, price_data):
        ret = calculate_log_returns(price_data, freq=1)
        expected_first = np.log(price_data.iloc[1]) - np.log(price_data.iloc[0])
        np.testing.assert_allclose(
            ret.iloc[0].values, expected_first.values, atol=1e-12
        )

    def test_abs_returns_shape(self, price_data):
        ret = compute_absolute_returns(price_data, freq=1)
        assert ret.shape == (59, 3)

    def test_abs_returns_values(self, price_data):
        ret = compute_absolute_returns(price_data, freq=1)
        expected_first = price_data.iloc[1] - price_data.iloc[0]
        np.testing.assert_allclose(
            ret.iloc[0].values, expected_first.values, atol=1e-12
        )

    def test_calculate_returns_dict_keys(self, returns_dict):
        expected_keys = {
            "return_type",
            "freq",
            "returns",
            "regime",
            "dates",
            "mean",
            "covariance",
            "tickers",
        }
        assert expected_keys == set(returns_dict.keys())

    def test_calculate_returns_mean_shape(self, returns_dict):
        assert returns_dict["mean"].shape == (3,)

    def test_calculate_returns_covariance_shape(self, returns_dict):
        assert returns_dict["covariance"].shape == (3, 3)

    def test_covariance_is_symmetric(self, returns_dict):
        cov = returns_dict["covariance"]
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)


# ---------------------------------------------------------------------------
# CvarData / CvarParameters
# ---------------------------------------------------------------------------


class TestCvarData:
    def test_shape(self, cvar_data):
        assert cvar_data.R.shape == (3, 200)
        assert cvar_data.mean.shape == (3,)
        assert cvar_data.p.shape == (200,)

    def test_probabilities_sum_to_one(self, cvar_data):
        np.testing.assert_allclose(cvar_data.p.sum(), 1.0, atol=1e-12)

    def test_uniform_probabilities(self, cvar_data):
        np.testing.assert_allclose(cvar_data.p, 1.0 / 200, atol=1e-12)


class TestCvarParameters:
    def test_requires_explicit_weight_bounds(self):
        # CVaR has no silent default bounds; constructing without w_min/w_max
        # must fail fast with a clear error (not infeasible numbers).
        with pytest.raises(ValueError):
            CvarParameters()

    def test_defaults(self):
        params = CvarParameters(w_min=0.0, w_max=1.0)
        assert params.confidence == 0.95
        assert params.cardinality is None
        assert params.T_tar is None

    def test_update_confidence_valid(self):
        params = CvarParameters(w_min=0.0, w_max=1.0)
        params.update_confidence(0.99)
        assert params.confidence == 0.99

    def test_update_confidence_invalid(self):
        params = CvarParameters(w_min=0.0, w_max=1.0)
        with pytest.raises(ValueError):
            params.update_confidence(0.0)
        with pytest.raises(ValueError):
            params.update_confidence(1.5)

    def test_update_risk_aversion_invalid(self):
        params = CvarParameters(w_min=0.0, w_max=1.0)
        with pytest.raises(ValueError):
            params.update_risk_aversion(-1.0)

    def test_update_cardinality_invalid(self):
        params = CvarParameters(w_min=0.0, w_max=1.0)
        with pytest.raises(ValueError):
            params.update_cardinality(-3)

    def test_update_c_min_invalid(self):
        params = CvarParameters(w_min=0.0, w_max=1.0)
        with pytest.raises(ValueError):
            params.update_c_min(-0.1)


# ---------------------------------------------------------------------------
# MeanVarianceParameters
# ---------------------------------------------------------------------------


class TestMeanVarianceParameters:
    def test_var_limit_must_be_positive(self):
        with pytest.raises(ValidationError):
            MeanVarianceParameters(var_limit=0.0)
        with pytest.raises(ValidationError):
            MeanVarianceParameters(var_limit=-1.0)

    def test_positive_var_limit_valid(self):
        params = MeanVarianceParameters(var_limit=0.01)
        assert params.var_limit == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Portfolio math
# ---------------------------------------------------------------------------


class TestPortfolio:
    def test_expected_return(self):
        mean = np.array([0.05, 0.10, 0.15])
        p = Portfolio(tickers=TICKERS, weights=np.array([0.3, 0.3, 0.4]))
        ret = p.calculate_portfolio_expected_return(mean)
        expected = 0.3 * 0.05 + 0.3 * 0.10 + 0.4 * 0.15
        np.testing.assert_allclose(ret, expected, atol=1e-12)

    def test_portfolio_variance(self):
        cov = np.array([[0.04, 0.01, 0.02], [0.01, 0.09, 0.03], [0.02, 0.03, 0.16]])
        w = np.array([0.5, 0.3, 0.2])
        p = Portfolio(tickers=TICKERS, weights=w)
        var = p.calculate_portfolio_variance(cov)
        expected = w @ cov @ w
        np.testing.assert_allclose(var, expected, atol=1e-12)

    def test_portfolio_from_dict(self):
        p = Portfolio(tickers=TICKERS)
        p.portfolio_from_dict("test", {"AAPL": 0.5, "GOOGL": 0.3}, 0.2)
        assert p.name == "test"
        np.testing.assert_allclose(p.weights, [0.5, 0.3, 0.0], atol=1e-12)
        assert p.cash == pytest.approx(0.2)

    def test_self_financing(self):
        p = Portfolio(tickers=TICKERS, weights=np.array([0.3, 0.3, 0.2]), cash=0.2)
        p._check_self_financing()  # should not raise

    def test_equality(self):
        p1 = Portfolio(tickers=TICKERS, weights=np.array([0.5, 0.3, 0.2]))
        p2 = Portfolio(tickers=TICKERS, weights=np.array([0.5, 0.3, 0.2]))
        assert p1 == p2

    def test_inequality(self):
        p1 = Portfolio(tickers=TICKERS, weights=np.array([0.5, 0.3, 0.2]))
        p2 = Portfolio(tickers=TICKERS, weights=np.array([0.1, 0.1, 0.8]))
        assert not (p1 == p2)


# ---------------------------------------------------------------------------
# CVaR computation
# ---------------------------------------------------------------------------


class TestComputeCVaR:
    def test_known_cvar(self):
        np.random.seed(99)
        n_scen = 10000
        R = np.random.randn(1, n_scen) * 0.02
        p = np.ones(n_scen) / n_scen
        data = CvarData(mean=np.array([0.0]), R=R, p=p)
        weights = np.array([1.0])

        cvar_95 = compute_CVaR(data, weights, confidence_level=0.95)
        assert cvar_95 > 0, "CVaR should be positive for a risky asset"

        cvar_99 = compute_CVaR(data, weights, confidence_level=0.99)
        assert cvar_99 > cvar_95, "99% CVaR should exceed 95% CVaR"

    def test_zero_weight_zero_cvar(self):
        R = np.array([[0.01, -0.02, 0.03], [0.02, -0.01, 0.01]])
        p = np.ones(3) / 3
        data = CvarData(mean=np.zeros(2), R=R, p=p)
        weights = np.array([0.0, 0.0])

        cvar = compute_CVaR(data, weights, confidence_level=0.95)
        assert cvar == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Normalize weights
# ---------------------------------------------------------------------------


class TestNormalizeWeights:
    def test_sums_to_one(self):
        weights_dict = {"AAPL": 0.3, "GOOGL": 0.4, "MSFT": 0.2}
        cash = 0.2
        nw, nc = normalize_portfolio_weights_to_one(weights_dict, cash)
        total = sum(nw.values()) + nc
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_preserves_ratios(self):
        weights_dict = {"A": 2.0, "B": 1.0}
        nw, _ = normalize_portfolio_weights_to_one(weights_dict, 0.0)
        assert nw["A"] / nw["B"] == pytest.approx(2.0, abs=1e-10)

    def test_already_normalized(self):
        weights_dict = {"A": 0.6, "B": 0.3}
        nw, nc = normalize_portfolio_weights_to_one(weights_dict, 0.1)
        np.testing.assert_allclose(nw["A"], 0.6, atol=1e-10)
        np.testing.assert_allclose(nc, 0.1, atol=1e-10)


# ---------------------------------------------------------------------------
# Scenario generation (generate_cvar_data)
# ---------------------------------------------------------------------------


class TestGenerateCvarData:
    def test_gaussian_fit(self, returns_dict):
        np.random.seed(7)
        rd = generate_cvar_data(
            returns_dict,
            ScenarioGenerationSettings(num_scen=100, fit_type="gaussian"),
        )
        cd = rd["cvar_data"]
        assert cd.R.shape == (3, 100)
        np.testing.assert_allclose(cd.p.sum(), 1.0, atol=1e-12)

    def _gaussian_R(self, returns_dict, seed):
        rd = generate_cvar_data(
            returns_dict,
            ScenarioGenerationSettings(num_scen=100, fit_type="gaussian", seed=seed),
        )
        return rd["cvar_data"].R

    def test_seed_makes_gaussian_scenarios_reproducible(self, returns_dict):
        first = self._gaussian_R(returns_dict, seed=1234)
        second = self._gaussian_R(returns_dict, seed=1234)
        np.testing.assert_array_equal(first, second)

    def test_different_seeds_give_different_gaussian_scenarios(self, returns_dict):
        assert not np.array_equal(
            self._gaussian_R(returns_dict, seed=1234),
            self._gaussian_R(returns_dict, seed=5678),
        )

    def test_unseeded_gaussian_scenarios_are_not_reproducible(self, returns_dict):
        """seed=None must keep drawing fresh entropy (the pre-existing default)."""
        assert not np.array_equal(
            self._gaussian_R(returns_dict, seed=None),
            self._gaussian_R(returns_dict, seed=None),
        )

    def test_seed_is_independent_of_global_numpy_state(self, returns_dict):
        """A seeded run must not be perturbed by the global RNG."""
        np.random.seed(1)
        first = self._gaussian_R(returns_dict, seed=99)
        np.random.seed(2)
        np.random.random(50)  # advance the global stream
        second = self._gaussian_R(returns_dict, seed=99)
        np.testing.assert_array_equal(first, second)

    def test_seed_makes_kde_scenarios_reproducible(self, returns_dict):
        settings = ScenarioGenerationSettings(
            num_scen=50,
            fit_type="kde",
            kde_settings=KDESettings(bandwidth=0.01, device="CPU"),
            seed=2024,
        )
        first = generate_cvar_data(returns_dict, settings)["cvar_data"].R
        second = generate_cvar_data(returns_dict, settings)["cvar_data"].R
        np.testing.assert_array_equal(first, second)

    def test_no_fit(self, returns_dict):
        rd = generate_cvar_data(
            returns_dict,
            ScenarioGenerationSettings(fit_type="no_fit"),
        )
        cd = rd["cvar_data"]
        n_obs = returns_dict["returns"].shape[0]
        assert cd.R.shape == (3, n_obs)

    def test_no_fit_returns_ndarray(self, returns_dict):
        """R must be an ndarray, not the DataFrame np.transpose would yield."""
        rd = generate_cvar_data(
            returns_dict,
            ScenarioGenerationSettings(fit_type="no_fit"),
        )
        assert isinstance(rd["cvar_data"].R, np.ndarray)

    def test_no_fit_uses_historical_returns_verbatim(self, returns_dict):
        """no_fit must pass the input through untouched, transposed."""
        expected = np.asarray(returns_dict["returns"]).T
        rd = generate_cvar_data(
            returns_dict,
            ScenarioGenerationSettings(fit_type="no_fit"),
        )
        np.testing.assert_allclose(rd["cvar_data"].R, expected, atol=1e-12)

    def test_no_fit_probabilities_are_uniform(self, returns_dict):
        rd = generate_cvar_data(
            returns_dict,
            ScenarioGenerationSettings(fit_type="no_fit"),
        )
        cd = rd["cvar_data"]
        n_obs = returns_dict["returns"].shape[0]
        # num_scen is derived from the data, not from the configured value
        assert cd.p.shape == (n_obs,)
        np.testing.assert_allclose(cd.p, 1.0 / n_obs, atol=1e-12)

    def test_invalid_fit_type(self):
        with pytest.raises(ValidationError):
            ScenarioGenerationSettings(num_scen=50, fit_type="magic")


class TestJointFMSettings:
    def test_jointfm_fit_type_gets_default_settings(self):
        settings = ScenarioGenerationSettings(fit_type="jointfm")
        assert settings.jointfm_settings == JointFMSettings()
        assert settings.jointfm_settings.expected_return == "scenario"

    def test_explicit_jointfm_settings_are_kept(self):
        jointfm_settings = JointFMSettings(
            expected_return="historical", dotenv_path=None, config_path=None
        )
        settings = ScenarioGenerationSettings(
            fit_type="jointfm", jointfm_settings=jointfm_settings
        )
        assert settings.jointfm_settings == jointfm_settings

    def test_other_fit_types_get_no_jointfm_settings(self):
        settings = ScenarioGenerationSettings(fit_type="kde")
        assert settings.jointfm_settings is None

    def test_invalid_expected_return(self):
        with pytest.raises(ValidationError):
            JointFMSettings(expected_return="forecast")


class TestJointFMExtra:
    def test_missing_extra_raises_install_hint(self, monkeypatch):
        # A None entry in sys.modules makes the import fail as if uninstalled.
        monkeypatch.setitem(sys.modules, "jointfm_client", None)
        with pytest.raises(ImportError, match="--extra jointfm") as excinfo:
            _import_jointfm_client()
        assert str(excinfo.value) == JOINTFM_INSTALL_HINT
        assert isinstance(excinfo.value.__cause__, ImportError)

    def test_installed_extra_is_imported(self):
        jointfm_client = pytest.importorskip("jointfm_client")
        assert _import_jointfm_client() is jointfm_client


class TestJointFMSimulator:
    def _settings(self, **kwargs):
        return ScenarioGenerationSettings(
            num_scen=25, fit_type="jointfm", seed=11, **kwargs
        )

    def test_scenarios_feed_cvar_data(self, returns_dict, fake_jointfm):
        cd = generate_cvar_data(returns_dict, self._settings())["cvar_data"]
        assert cd.R.shape == (3, 25)
        np.testing.assert_allclose(cd.p, np.full(25, 1 / 25))
        # The default takes expected returns from the scenarios themselves.
        np.testing.assert_allclose(cd.mean, cd.R.mean(axis=1))

    def test_historical_expected_returns_keep_the_scenarios(
        self, returns_dict, fake_jointfm
    ):
        historical = generate_cvar_data(
            dict(returns_dict),
            self._settings(
                jointfm_settings=JointFMSettings(expected_return="historical")
            ),
        )["cvar_data"]
        scenario = generate_cvar_data(dict(returns_dict), self._settings())["cvar_data"]
        np.testing.assert_array_equal(historical.R, scenario.R)
        np.testing.assert_array_equal(historical.mean, returns_dict["mean"])
        np.testing.assert_array_equal(
            expected_returns(returns_dict, self._settings(), scenario.R.T),
            scenario.mean,
        )

    def test_one_joint_next_step_request(self, returns_dict, fake_jointfm):
        generate_cvar_data(returns_dict, self._settings())
        (call,) = fake_jointfm.forecast_calls
        assert call["n_samples"] == 25
        assert call["seed"] == 11
        assert call["return_mode"] == "samples"
        assert call["target_columns"] == TICKERS
        assert call["requested_columns"] == TICKERS
        assert call["query_times"] == [len(call["history"])]

    @staticmethod
    def _assert_log_prices_up_to_constant(history, prices):
        """The history is the log of ``prices`` shifted by one constant per asset."""
        assert list(history.columns) == list(prices.columns)
        offsets = history.to_numpy() - np.log(prices.to_numpy())
        np.testing.assert_allclose(
            offsets, np.broadcast_to(offsets[0], offsets.shape), rtol=0, atol=1e-12
        )

    @pytest.mark.parametrize("return_type", ["LOG", "LINEAR"])
    def test_history_is_log_price_path(self, price_data, fake_jointfm, return_type):
        returns_dict = calculate_returns(
            price_data,
            regime_dict=None,
            returns_compute_settings=ReturnsComputeSettings(
                return_type=return_type, freq=1
            ),
        )
        fake_jointfm.envelope.n_input = 1000
        generate_cvar_data(returns_dict, self._settings())
        history = fake_jointfm.forecast_calls[0]["history"]
        self._assert_log_prices_up_to_constant(history, price_data)
        # The path starts from 0 at the price before the first return.
        np.testing.assert_array_equal(history.iloc[0], 0.0)

    def test_history_cut_to_n_input(self, price_data, returns_dict, fake_jointfm):
        n_input = fake_jointfm.envelope.n_input
        assert len(returns_dict["returns"]) > n_input
        generate_cvar_data(returns_dict, self._settings())
        history = fake_jointfm.forecast_calls[0]["history"]
        self._assert_log_prices_up_to_constant(history, price_data.iloc[-n_input:])

    @pytest.mark.parametrize("return_type", ["LOG", "LINEAR"])
    def test_forecast_log_price_becomes_next_return(
        self, returns_dict, fake_jointfm, return_type
    ):
        samples = generate_samples_jointfm(
            25, returns_dict["returns"], return_type, 1, JointFMSettings(), seed=3
        )
        (call,) = fake_jointfm.forecast_calls
        last_log_prices = call["history"].iloc[-1].to_numpy()
        assert np.all(np.abs(last_log_prices) > 1e-3)
        next_log_returns = call["samples"][:, 0, :] - last_log_prices
        expected = (
            next_log_returns if return_type == "LOG" else np.expm1(next_log_returns)
        )
        np.testing.assert_allclose(samples, expected)
        assert samples.flags.writeable

    @pytest.mark.parametrize("return_type", ["LOG", "LINEAR"])
    def test_horizon_samples_are_returns_over_each_horizon(
        self, returns_dict, fake_jointfm, return_type
    ):
        """Each horizon's draw minus the last log price is its h-period return."""
        samples = generate_horizon_samples_jointfm(
            25,
            returns_dict["returns"],
            return_type,
            1,
            JointFMSettings(),
            (1, 5, 21),
            seed=3,
        )
        (call,) = fake_jointfm.forecast_calls
        last = len(call["history"]) - 1
        assert call["query_times"] == [last + 1, last + 5, last + 21]
        log_returns = call["samples"] - call["history"].iloc[-1].to_numpy()
        expected = log_returns if return_type == "LOG" else np.expm1(log_returns)
        assert samples.shape == (25, 3, len(TICKERS))
        np.testing.assert_allclose(samples, expected)

    def test_horizon_beyond_n_output_is_refused(self, returns_dict, fake_jointfm):
        """A horizon past the served model's n_output is refused before sending."""
        fake_jointfm.envelope.n_output = 20
        with pytest.raises(ValueError, match="n_output=20"):
            generate_horizon_samples_jointfm(
                25, returns_dict["returns"], "LOG", 1, JointFMSettings(), (1, 21)
            )
        assert fake_jointfm.forecast_calls == []

    @pytest.mark.parametrize("horizons", [(), (0, 1), (5, 5), (5, 1)])
    def test_invalid_horizons_are_refused(self, returns_dict, fake_jointfm, horizons):
        """Empty, non-positive, or non-increasing horizons never reach the client."""
        with pytest.raises(ValueError, match="strictly increasing"):
            generate_horizon_samples_jointfm(
                25, returns_dict["returns"], "LOG", 1, JointFMSettings(), horizons
            )
        assert fake_jointfm.from_env_kwargs is None

    def test_scenarios_never_imply_non_positive_prices(
        self, returns_dict, fake_jointfm
    ):
        """A strongly negative forecast log return still maps to a price above 0."""
        returns_dict["return_type"] = "LINEAR"
        returns_dict["returns"] = np.expm1(returns_dict["returns"])
        fake_jointfm.sample_scale = 5.0
        cd = generate_cvar_data(returns_dict, self._settings())["cvar_data"]
        assert cd.R.min() < -0.9
        assert np.all(1.0 + cd.R > 0.0)

    @pytest.mark.parametrize("return_type", ["ABSOLUTE", "PNL"])
    def test_return_type_without_price_level_is_refused(
        self, returns_dict, fake_jointfm, return_type
    ):
        returns_dict["return_type"] = return_type
        with pytest.raises(ValueError, match="return_type"):
            generate_cvar_data(returns_dict, self._settings())
        assert fake_jointfm.from_env_kwargs is None

    def test_overlapping_returns_are_refused(self, returns_dict, fake_jointfm):
        returns_dict["freq"] = 5
        with pytest.raises(ValueError, match="freq=5"):
            generate_cvar_data(returns_dict, self._settings())
        assert fake_jointfm.from_env_kwargs is None

    def test_client_reads_settings_environment(self, returns_dict, fake_jointfm):
        settings = self._settings(
            jointfm_settings=JointFMSettings(dotenv_path=None, config_path="c.yaml")
        )
        generate_cvar_data(returns_dict, settings)
        assert fake_jointfm.from_env_kwargs == {
            "dotenv_path": None,
            "config_path": "c.yaml",
        }

    @pytest.mark.parametrize("min_series,max_series", [(1, 2), (4, 10)])
    def test_universe_outside_capacity_is_refused(
        self, returns_dict, fake_jointfm, min_series, max_series
    ):
        fake_jointfm.envelope.min_series = min_series
        fake_jointfm.envelope.max_series = max_series
        with pytest.raises(ValueError, match="got 3 assets"):
            generate_cvar_data(returns_dict, self._settings())
        assert fake_jointfm.forecast_calls == []

    def test_missing_capacity_envelope_is_refused(self, returns_dict, fake_jointfm):
        fake_jointfm.envelope = None
        with pytest.raises(ValueError, match="data_generation"):
            generate_cvar_data(returns_dict, self._settings())
        assert fake_jointfm.forecast_calls == []

    def test_scenario_expected_return_leaves_returns_dict_mean(
        self, returns_dict, fake_jointfm
    ):
        historical_mean = returns_dict["mean"].copy()
        cd = generate_cvar_data(returns_dict, self._settings())["cvar_data"]
        assert not np.allclose(cd.mean, historical_mean)
        np.testing.assert_array_equal(returns_dict["mean"], historical_mean)

    def test_historical_expected_return_is_window_mean(
        self, returns_dict, fake_jointfm
    ):
        settings = self._settings(
            jointfm_settings=JointFMSettings(expected_return="historical")
        )
        cd = generate_cvar_data(returns_dict, settings)["cvar_data"]
        np.testing.assert_array_equal(cd.mean, returns_dict["mean"])

    def _rebalancer(
        self,
        tmp_path,
        cvar_params,
        returns_compute_settings,
        losing=False,
        api_settings=None,
        # Any period return is below +100%, so every period re-optimizes.
        re_optimize_criteria=(("type", "pct_change"), ("threshold", 1.0)),
    ):
        dates = pd.date_range("2023-01-02", periods=200, freq="B")
        if losing:
            price_frame = _losing_price_frame(dates)
        else:
            # Positive drift keeps the default risk-aversion scaling defined in
            # every look-back window.
            rng = np.random.default_rng(5)
            log_returns = rng.normal(0.003, 0.01, (200, 3))
            prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
            price_frame = pd.DataFrame(prices, index=dates, columns=TICKERS)
        price_path = tmp_path / "prices.csv"
        price_frame.to_csv(price_path)

        return rebalance.rebalance_portfolio(
            dataset_directory=str(price_path),
            returns_compute_settings=returns_compute_settings,
            # The price frames set the sign of the historical mean, which the
            # risk-aversion scaling reads; the fake client's scenario mean is noise.
            scenario_generation_settings=self._settings(
                jointfm_settings=JointFMSettings(expected_return="historical")
            ),
            trading_start="2023-05-01",
            trading_end="2023-08-31",
            look_forward_window=10,
            look_back_window=60,
            cvar_params=cvar_params,
            solver_settings={"solver": cp.CLARABEL, "verbose": False},
            re_optimize_criteria=dict(re_optimize_criteria),
            api_settings=api_settings,
        )

    def test_every_step_re_optimizes_after_every_window(
        self, tmp_path, fake_jointfm, cvar_params
    ):
        rebalancer = self._rebalancer(
            tmp_path,
            cvar_params,
            ReturnsComputeSettings(return_type="LOG", freq=1),
            re_optimize_criteria=(("type", "every_step"), ("threshold", None)),
        )
        _, re_optimize_dates, _ = rebalancer.re_optimize()
        # The first window holds the buy-and-hold portfolio; each later one starts
        # with a re-optimization, up to the last complete window.
        window = rebalancer.look_forward_window
        last_start = len(rebalancer.dates_range) - window
        assert re_optimize_dates == list(
            rebalancer.dates_range[window:last_start:window]
        )

    def test_unknown_re_optimization_type_is_refused(
        self, tmp_path, fake_jointfm, cvar_params
    ):
        with pytest.raises(ValueError, match="Unknown re-optimization type 'monthly'"):
            self._rebalancer(
                tmp_path,
                cvar_params,
                ReturnsComputeSettings(return_type="LOG", freq=1),
                re_optimize_criteria=(("type", "monthly"), ("threshold", 0.0)),
            )

    def test_rebalancing_applies_api_settings_to_every_optimization(
        self, tmp_path, fake_jointfm, cvar_params
    ):
        """Unscaled risk aversion carries a rebalancing run through losing windows."""
        returns_compute_settings = ReturnsComputeSettings(return_type="LOG", freq=1)
        with pytest.raises(ValueError, match="scale_risk_aversion=False"):
            self._rebalancer(
                tmp_path, cvar_params, returns_compute_settings, losing=True
            )

        rebalancer = self._rebalancer(
            tmp_path,
            cvar_params,
            returns_compute_settings,
            losing=True,
            api_settings=ApiSettings(scale_risk_aversion=False),
        )
        _, re_optimize_dates, _ = rebalancer.re_optimize()
        assert len(re_optimize_dates) > 1

    def test_rebalancing_plot_names_scenario_generator(
        self, tmp_path, fake_jointfm, cvar_params
    ):
        """Title and file name say which scenario generator drove the rebalancing."""
        returns_compute_settings = ReturnsComputeSettings(return_type="LOG", freq=1)
        rebalancer = self._rebalancer(tmp_path, cvar_params, returns_compute_settings)
        results_dir = tmp_path / "plots"
        rebalancer.re_optimize(
            plot_results=True, save_plot=True, results_dir=str(results_dir)
        )

        title = plt.gcf().axes[0].get_title()
        plt.close("all")
        assert title.strip() == (
            "Percentage Change Rebalancing Strategy with JointFM Scenarios"
        )
        (saved_plot,) = results_dir.iterdir()
        assert saved_plot.name.startswith("rebalancing_jointfm_pct-change_")

    def test_rebalancing_forecasts_each_look_back_window(
        self, tmp_path, fake_jointfm, cvar_params
    ):
        """Every rebalance sends the log-price path of that date's look-back window."""
        returns_compute_settings = ReturnsComputeSettings(return_type="LOG", freq=1)
        rebalancer = self._rebalancer(tmp_path, cvar_params, returns_compute_settings)
        _, re_optimize_dates, _ = rebalancer.re_optimize()

        # One buy-and-hold optimization at trading_start, then one per rebalance.
        optimize_dates = [pd.Timestamp("2023-05-01"), *re_optimize_dates]
        assert len(re_optimize_dates) > 1
        assert len(fake_jointfm.forecast_calls) == len(optimize_dates)
        n_input = fake_jointfm.envelope.n_input
        for call, optimize_date in zip(
            fake_jointfm.forecast_calls, optimize_dates, strict=True
        ):
            window_range = (
                (optimize_date - pd.Timedelta(days=60)).strftime("%Y-%m-%d"),
                optimize_date.strftime("%Y-%m-%d"),
            )
            window = utils.calculate_returns(
                rebalancer.price_data,
                {"name": "re-optimize", "range": window_range},
                returns_compute_settings,
            )["returns"]
            window_prices = rebalancer.price_data.loc[slice(*window_range)]
            assert len(window_prices) == len(window) + 1 > n_input
            assert window.index[-1] == window_prices.index[-1] <= optimize_date
            self._assert_log_prices_up_to_constant(
                call["history"], window_prices.iloc[-n_input:]
            )


# ---------------------------------------------------------------------------
# Small CVaR optimization (CVXPY + CLARABEL, CPU only)
# ---------------------------------------------------------------------------


class TestCVaROptimization:
    def test_basic_optimization_feasible(self, returns_dict, cvar_data, cvar_params):
        returns_dict["cvar_data"] = cvar_data
        optimizer = CVaR(returns_dict=returns_dict, cvar_params=cvar_params)
        result, portfolio = optimizer.solve_optimization_problem(
            {"solver": cp.CLARABEL, "verbose": False}, print_results=False
        )

        assert portfolio is not None
        w = portfolio.weights
        c = portfolio.cash

        np.testing.assert_allclose(np.sum(w) + c, 1.0, atol=1e-4)

        assert np.all(w >= -1e-4), "weights should respect lower bound"
        assert np.all(w <= 0.5 + 1e-4), "weights should respect upper bound"

    def test_optimization_returns_expected_keys(
        self, returns_dict, cvar_data, cvar_params
    ):
        returns_dict["cvar_data"] = cvar_data
        optimizer = CVaR(returns_dict=returns_dict, cvar_params=cvar_params)
        result, _ = optimizer.solve_optimization_problem(
            {"solver": cp.CLARABEL, "verbose": False}, print_results=False
        )

        for key in ["regime", "solver", "solve time", "return", "CVaR", "obj"]:
            assert key in result.index, f"Missing key: {key}"

    def test_cvar_consistent_with_compute(self, returns_dict, cvar_data, cvar_params):
        returns_dict["cvar_data"] = cvar_data
        optimizer = CVaR(returns_dict=returns_dict, cvar_params=cvar_params)
        result, portfolio = optimizer.solve_optimization_problem(
            {"solver": cp.CLARABEL, "verbose": False}, print_results=False
        )

        reported_cvar = result["CVaR"]
        computed_cvar = compute_CVaR(
            cvar_data, portfolio.weights, cvar_params.confidence
        )

        np.testing.assert_allclose(reported_cvar, computed_cvar, atol=0.02)

    def test_risk_aversion_scaling_refuses_window_without_positive_return(
        self, losing_returns_dict, cvar_params
    ):
        """Scaling by a non-positive return/CVaR ratio has no meaning, so it raises."""
        with pytest.raises(ValueError, match="scale_risk_aversion=False"):
            CVaR(returns_dict=losing_returns_dict, cvar_params=cvar_params)

    def test_unscaled_risk_aversion_solves_window_without_positive_return(
        self, losing_returns_dict, cvar_params
    ):
        optimizer = CVaR(
            returns_dict=losing_returns_dict,
            cvar_params=cvar_params,
            api_settings=ApiSettings(scale_risk_aversion=False),
        )
        assert optimizer.params.risk_aversion == cvar_params.risk_aversion
        _, portfolio = optimizer.solve_optimization_problem(
            {"solver": cp.CLARABEL, "verbose": False}, print_results=False
        )
        np.testing.assert_allclose(np.sum(portfolio.weights) + portfolio.cash, 1.0)


# ---------------------------------------------------------------------------
# Small Mean-Variance optimization and SOCP variance caps
# ---------------------------------------------------------------------------


class TestMeanVarianceOptimization:
    def _equal_weight_variance_cap(self, returns_dict, multiplier=1.05):
        covariance = np.asarray(returns_dict["covariance"], dtype=float)
        weights = np.ones(len(returns_dict["tickers"])) / len(returns_dict["tickers"])
        return float(weights @ covariance @ weights) * multiplier

    def test_cvxpy_var_limit_respects_cap(self, returns_dict):
        variance_cap = self._equal_weight_variance_cap(returns_dict)
        params = MeanVarianceParameters(
            w_min=0.0,
            w_max=1.0,
            c_min=0.0,
            c_max=0.0,
            L_tar=1.0,
            var_limit=variance_cap,
        )
        optimizer = mean_variance_optimizer.MeanVariance(
            returns_dict=returns_dict,
            mean_variance_params=params,
        )
        result, portfolio = optimizer.solve_optimization_problem(
            {"solver": cp.CLARABEL, "verbose": False}, print_results=False
        )

        weights = np.asarray(portfolio.weights, dtype=float).flatten()
        cash = float(np.asarray(portfolio.cash).squeeze())
        realized_variance = float(weights @ returns_dict["covariance"] @ weights)

        np.testing.assert_allclose(np.sum(weights) + cash, 1.0, atol=1e-4)
        assert realized_variance <= variance_cap + 1e-6
        assert float(result["variance"]) <= variance_cap + 1e-6
        assert float(result["return"]) > -1.0

    def test_risk_aversion_scaling_refuses_window_without_positive_return(
        self, losing_returns_dict
    ):
        """Scaling by a non-positive return/std ratio has no meaning, so it raises."""
        with pytest.raises(ValueError, match="scale_risk_aversion=False"):
            mean_variance_optimizer.MeanVariance(
                returns_dict=losing_returns_dict,
                mean_variance_params=MeanVarianceParameters(),
            )

    @pytest.mark.gpu
    def test_cuopt_python_var_limit_solves_socp(self, returns_dict, require_cuopt_socp):
        variance_cap = self._equal_weight_variance_cap(returns_dict)
        params = MeanVarianceParameters(
            w_min=0.0,
            w_max=1.0,
            c_min=0.0,
            c_max=0.0,
            L_tar=1.0,
            var_limit=variance_cap,
        )
        optimizer = mean_variance_optimizer.MeanVariance(
            returns_dict=returns_dict,
            mean_variance_params=params,
            api_settings=ApiSettings(api="cuopt_python"),
        )
        result, portfolio = optimizer.solve_optimization_problem(print_results=False)

        weights = np.asarray(portfolio.weights, dtype=float).flatten()
        cash = float(np.asarray(portfolio.cash).squeeze())
        realized_variance = float(weights @ returns_dict["covariance"] @ weights)

        np.testing.assert_allclose(np.sum(weights) + cash, 1.0, atol=1e-4)
        assert str(result["solver"]).lower() == "cuopt"
        assert realized_variance <= variance_cap + 1e-6


# ---------------------------------------------------------------------------
# Efficient frontier end-to-end
# ---------------------------------------------------------------------------


class TestEfficientFrontier:
    def test_zero_volatility_sharpe_is_nan(self):
        assert np.isnan(_annualized_sharpe_ratio(0.0, 0.0))

    def test_key_portfolios_skip_origin_when_risky_portfolios_exist(self):
        results_df = pd.DataFrame(
            {
                "variance": [0.0, 0.01, 0.04],
                "return": [0.0, 0.06, 0.08],
                "sharpe": [np.nan, 0.95, 0.75],
            },
            index=["origin", "low_risk", "high_risk"],
        )

        key_portfolios = _select_efficient_frontier_key_portfolios(results_df)

        assert key_portfolios["Min Variance"] == "low_risk"
        assert key_portfolios["Max Sharpe"] == "low_risk"

    def test_frontier_monotonicity(self, returns_dict, cvar_data, cvar_params):
        returns_dict["cvar_data"] = cvar_data
        solver_settings = {"solver": cp.CLARABEL, "verbose": False}

        results_df, fig, ax = create_efficient_frontier(
            returns_dict,
            cvar_params,
            solver_settings,
            ra_num=3,
            min_risk_aversion=-1,
            max_risk_aversion=1,
            show_plot=False,
            show_discretized_portfolios=False,
            benchmark_portfolios=False,
            print_portfolio_results=False,
        )

        assert len(results_df) == 3
        assert "return" in results_df.columns
        assert "CVaR" in results_df.columns
        assert "variance" in results_df.columns

        sorted_by_risk = results_df.sort_values("CVaR")
        returns_sorted = sorted_by_risk["return"].values
        assert returns_sorted[-1] >= returns_sorted[0] - 1e-6, (
            "higher risk should generally yield higher return on the frontier"
        )

        for _, row in results_df.iterrows():
            assert row["variance"] >= 0, "variance must be non-negative"
            assert np.isfinite(row["sharpe"]), "sharpe should be finite"

    def test_frontier_exposes_per_asset_weights(
        self, returns_dict, cvar_data, cvar_params
    ):
        returns_dict["cvar_data"] = cvar_data
        results_df, _fig, _ax = create_efficient_frontier(
            returns_dict,
            cvar_params,
            {"solver": cp.CLARABEL, "verbose": False},
            ra_num=3,
            min_risk_aversion=-1,
            max_risk_aversion=1,
            show_plot=False,
            show_discretized_portfolios=False,
            benchmark_portfolios=False,
            print_portfolio_results=False,
        )
        assert "weights" in results_df.columns
        assert "cash" in results_df.columns
        first = results_df["weights"].iloc[0]
        assert isinstance(first, dict)
        assert set(first.keys()) == set(TICKERS)

    def test_discretized_overlay_runs(self, returns_dict, cvar_data, cvar_params):
        # Regression: show_discretized_portfolios=True with the default
        # discretization_params used to raise TypeError via a stale 'sum_to_one'
        # kwarg. It must now run end-to-end (NumPy fallback on CPU).
        returns_dict["cvar_data"] = cvar_data
        results_df, _fig, _ax = create_efficient_frontier(
            returns_dict,
            cvar_params,
            {"solver": cp.CLARABEL, "verbose": False},
            ra_num=2,
            min_risk_aversion=-1,
            max_risk_aversion=1,
            show_plot=False,
            show_discretized_portfolios=True,
            benchmark_portfolios=False,
            print_portfolio_results=False,
        )
        assert len(results_df) == 2


# ---------------------------------------------------------------------------
# Backtester with canned data
# ---------------------------------------------------------------------------


class TestBacktester:
    @pytest.fixture()
    def backtest_returns_dict(self):
        np.random.seed(123)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        daily_returns = np.random.randn(252, 3) * 0.01 + 0.0003
        returns_df = pd.DataFrame(daily_returns, index=dates, columns=TICKERS)
        return {
            "return_type": "LOG",
            "freq": 1,
            "returns": returns_df,
            "dates": returns_df.index,
            "mean": np.mean(daily_returns, axis=0),
            "covariance": np.cov(daily_returns.T),
            "tickers": TICKERS,
        }

    def test_sharpe_ratio_positive_for_positive_drift(self, backtest_returns_dict):
        portfolio = Portfolio(
            name="test",
            tickers=TICKERS,
            weights=np.array([0.4, 0.3, 0.3]),
            cash=0.0,
        )
        bt = portfolio_backtester(
            test_portfolio=portfolio,
            returns_dict=backtest_returns_dict,
            risk_free_rate=0.0,
            test_method="historical",
        )
        results, _ = bt.backtest_against_benchmarks(plot_returns=False)
        sharpe = results.loc["test", "sharpe"]
        assert sharpe > 0, "positive drift data should produce positive Sharpe"

    def test_sortino_exceeds_sharpe(self, backtest_returns_dict):
        portfolio = Portfolio(
            name="test",
            tickers=TICKERS,
            weights=np.array([0.4, 0.3, 0.3]),
            cash=0.0,
        )
        bt = portfolio_backtester(
            test_portfolio=portfolio,
            returns_dict=backtest_returns_dict,
            risk_free_rate=0.0,
            test_method="historical",
        )
        results, _ = bt.backtest_against_benchmarks(plot_returns=False)
        sharpe = results.loc["test", "sharpe"]
        sortino = results.loc["test", "sortino"]
        assert sortino > sharpe, (
            "Sortino should exceed Sharpe when downside vol < total vol"
        )

    def test_max_drawdown_bounded(self, backtest_returns_dict):
        portfolio = Portfolio(
            name="test",
            tickers=TICKERS,
            weights=np.array([0.4, 0.3, 0.3]),
            cash=0.0,
        )
        bt = portfolio_backtester(
            test_portfolio=portfolio,
            returns_dict=backtest_returns_dict,
            risk_free_rate=0.0,
            test_method="historical",
        )
        results, _ = bt.backtest_against_benchmarks(plot_returns=False)
        mdd = results.loc["test", "max drawdown"]
        assert 0 <= mdd <= 1, "max drawdown should be between 0 and 1"

    def test_cumulative_returns_anchor_to_regime_start(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        prices = pd.DataFrame(
            {
                "AAPL": [100.0, 101.0, 102.0, 103.0],
                "GOOGL": [50.0, 49.0, 50.0, 51.0],
                "MSFT": [200.0, 200.0, 202.0, 204.0],
            },
            index=dates,
        )
        returns_dict = calculate_returns(
            prices,
            regime_dict={"name": "test", "range": ("2024-01-02", "2024-01-05")},
            returns_compute_settings=ReturnsComputeSettings(
                return_type="LINEAR", freq=1
            ),
        )
        portfolio = Portfolio(
            name="test",
            tickers=TICKERS,
            weights=np.array([0.5, 0.5, 0.0]),
            cash=0.0,
        )
        bt = portfolio_backtester(
            test_portfolio=portfolio,
            returns_dict=returns_dict,
            risk_free_rate=0.0,
            benchmark_portfolios=[],
        )

        result = bt.backtest_single_portfolio(portfolio)
        cumulative_returns = result["cumulative returns"].iloc[0]

        assert bt.cumulative_dates[0] == pd.Timestamp("2024-01-02")
        assert bt.cumulative_dates[1] == pd.Timestamp("2024-01-03")
        assert len(cumulative_returns) == len(bt.cumulative_dates)
        assert cumulative_returns[0] == pytest.approx(1.0)
        assert cumulative_returns[1] == pytest.approx(0.995)

    def test_drawdown_known_series(self):
        values = np.array([1.0, 1.1, 1.05, 0.9, 0.95, 1.0])
        bt = portfolio_backtester.__new__(portfolio_backtester)
        mdd = bt.max_drawdown(values)
        expected = (1.1 - 0.9) / 1.1
        np.testing.assert_allclose(mdd, expected, atol=1e-10)


class TestCompareResultsLabels:
    """Output of compare_results must contain 'cuOpt' (mixed case) for the
    GPU solver regardless of whether it was invoked via CVXPY's `cp.CUOPT`
    (which str()'s to 'CUOPT') or the cuOpt Python API (which uses 'cuOpt').
    The QA-image notebook test parses the rendered output for the literal
    string 'cuOpt' followed by a solve time, so this is a contract."""

    @staticmethod
    def _row(solver, t=1.234):
        return pd.Series(
            {"regime": "r", "solver": solver, "solve time": t, "obj": -0.5}
        )

    def test_cvxpy_cuopt_displays_as_cuopt_gpu(self, capsys):
        compare_results(self._row("CUOPT"), self._row("CLARABEL", t=5.6))
        out = capsys.readouterr().out
        assert "cuOpt (GPU)" in out
        assert "CLARABEL (CPU)" in out
        # The QA pattern is roughly r"cuOpt.*?(\d+\.\d+)" - make sure it matches
        import re

        assert re.search(r"cuOpt.*?(\d+\.\d+)", out)

    def test_cuopt_python_api_displays_as_cuopt_gpu(self, capsys):
        compare_results(self._row("cuOpt"), self._row("CLARABEL", t=5.6))
        out = capsys.readouterr().out
        assert "cuOpt (GPU)" in out
        assert "CLARABEL (CPU)" in out

    def test_objective_diff_uses_friendly_labels(self, capsys):
        compare_results(self._row("CUOPT"), self._row("CLARABEL", t=5.6))
        out = capsys.readouterr().out
        assert "cuOpt (GPU) vs CLARABEL (CPU)" in out

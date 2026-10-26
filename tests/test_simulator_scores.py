# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scores of the simulator comparison in experiments/ on synthetic data whose
answers are known."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist
from scipy.stats import norm

from experiments.simulator_comparison.config import CvarConfig
from experiments.simulator_comparison.portfolios import (
    PORTFOLIO_STREAM,
    draw_portfolios,
)
from experiments.simulator_comparison.scores import (
    PIT_BINS,
    PORTFOLIO_CRPS,
    PORTFOLIO_SUBSETS,
    TRAILING_STD_WINDOW,
    PortfolioScores,
    ScenarioScores,
    cell_column,
    energy_score,
    interval_coverage,
    probability_integral_transform,
    recorded_cells,
    sample_crps,
    variogram_score,
)
from experiments.simulator_comparison.trials import OriginInputs


def _gaussian_crps(mean, std, realized):
    z = (realized - mean) / std
    return std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def _mean(window):
    """The window's mean return per asset, the expected returns most arms use."""
    return window.mean().to_numpy()


def _inputs(window, forward, seed=0):
    return OriginInputs(
        origin=window.index[-1],
        returns_dict={"returns": window},
        forward=forward,
        seed=seed,
    )


BOUNDS = CvarConfig(
    w_min=-0.3,
    w_max=0.8,
    c_min=0.1,
    c_max=0.2,
    L_tar=1.6,
    T_tar=0.5,
    confidence=0.95,
    risk_aversion=1.0,
)


@pytest.fixture()
def returns():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2021-01-01", periods=80, freq="B")
    values = rng.multivariate_normal(
        np.zeros(4), 0.0001 * (0.5 * np.eye(4) + 0.5), size=80
    )
    return pd.DataFrame(values, index=dates, columns=list("ABCD"))


class TestMarginalScores:
    def test_sample_crps_matches_the_gaussian_closed_form(self):
        rng = np.random.default_rng(0)
        mean, std = 0.3, 2.0
        realized = np.array([-3.0, 0.3, 1.5])
        samples = rng.normal(mean, std, size=(200_000, 3))
        np.testing.assert_allclose(
            sample_crps(samples, realized),
            _gaussian_crps(mean, std, realized),
            rtol=1e-2,
        )

    def test_matching_distribution_is_calibrated(self):
        rng = np.random.default_rng(1)
        assets = 4000
        scale = rng.uniform(0.5, 3.0, size=assets)
        samples = rng.normal(0.0, scale, size=(2000, assets))
        realized = rng.normal(0.0, scale)

        for level in (0.5, 0.9):
            coverage = interval_coverage(samples, realized, level).mean()
            assert coverage == pytest.approx(level, abs=0.03)
        shares, _ = np.histogram(
            probability_integral_transform(samples, realized),
            bins=PIT_BINS,
            range=(0.0, 1.0),
        )
        np.testing.assert_allclose(shares / assets, 1 / PIT_BINS, atol=0.015)

    def test_overconfident_forecast_covers_too_little(self):
        rng = np.random.default_rng(2)
        samples = rng.normal(0.0, 0.5, size=(2000, 4000))
        realized = rng.normal(0.0, 1.0, size=4000)
        assert interval_coverage(samples, realized, 0.9).mean() < 0.6

    def test_pit_counts_ties_half(self):
        samples = np.array([[0.0], [1.0], [1.0], [2.0]])
        np.testing.assert_allclose(
            probability_integral_transform(samples, np.array([1.0])), [0.5]
        )


class TestEnergyScore:
    def test_prefers_correct_dependence_over_equal_marginals(self):
        rng = np.random.default_rng(3)
        dimension, correlation = 5, 0.8
        covariance = (1 - correlation) * np.eye(dimension) + correlation
        outcomes = rng.multivariate_normal(np.zeros(dimension), covariance, size=300)
        correct = rng.multivariate_normal(np.zeros(dimension), covariance, size=500)
        independent = rng.normal(size=(500, dimension))

        def mean_score(samples):
            return np.mean(
                [energy_score(samples, y, subsample=500, rng=rng) for y in outcomes]
            )

        assert mean_score(correct) < mean_score(independent)

    def test_exact_for_the_empirical_distribution_within_the_subsample(self):
        rng = np.random.default_rng(4)
        samples = rng.normal(size=(50, 3))
        realized = rng.normal(size=3)
        accuracy = np.linalg.norm(samples - realized, axis=1).mean()
        spread = cdist(samples, samples).mean()
        assert energy_score(samples, realized, subsample=50, rng=rng) == pytest.approx(
            accuracy - spread / 2
        )

    def test_subsample_estimates_the_full_score_without_bias(self):
        rng = np.random.default_rng(5)
        samples = rng.normal(size=(600, 3))
        realized = rng.normal(size=3)
        full = energy_score(samples, realized, subsample=600, rng=rng)
        estimates = [
            energy_score(samples, realized, subsample=60, rng=rng) for _ in range(400)
        ]
        assert np.mean(estimates) == pytest.approx(full, abs=3e-3)


class TestVariogramScore:
    def test_matches_the_definition_on_a_small_sample(self):
        samples = np.array([[0.0, 1.0, 3.0], [2.0, 2.0, 0.0]])
        realized = np.array([1.0, 0.0, 4.0])
        pairs = [(0, 1), (0, 2), (1, 2)]
        expected = np.mean(
            [
                (
                    abs(realized[i] - realized[j]) ** 0.5
                    - np.mean(np.abs(samples[:, i] - samples[:, j]) ** 0.5)
                )
                ** 2
                for i, j in pairs
            ]
        )
        score = variogram_score(
            samples, realized, order=0.5, subsample=10, rng=np.random.default_rng(0)
        )
        assert score == pytest.approx(expected)

    def test_prefers_correct_dependence_over_equal_marginals(self):
        rng = np.random.default_rng(3)
        dimension, correlation = 5, 0.8
        covariance = (1 - correlation) * np.eye(dimension) + correlation
        outcomes = rng.multivariate_normal(np.zeros(dimension), covariance, size=300)
        correct = rng.multivariate_normal(np.zeros(dimension), covariance, size=500)
        independent = rng.normal(size=(500, dimension))

        def mean_score(samples):
            return np.mean(
                [
                    variogram_score(samples, y, order=0.5, subsample=500, rng=rng)
                    for y in outcomes
                ]
            )

        assert mean_score(correct) < mean_score(independent)

    def test_subsample_stays_close_to_the_full_score(self):
        rng = np.random.default_rng(5)
        samples = rng.normal(size=(4000, 6))
        realized = rng.normal(size=6)
        full = variogram_score(samples, realized, order=0.5, subsample=4000, rng=rng)
        part = variogram_score(samples, realized, order=0.5, subsample=1000, rng=rng)
        assert part == pytest.approx(full, rel=0.05)


class TestScenarioScores:
    def test_returns_every_column(self, returns):
        window, forward = returns.iloc[:70], returns.iloc[70:]
        rng = np.random.default_rng(6)
        samples = rng.multivariate_normal(
            np.zeros(4), window.cov().to_numpy(), size=500
        )
        scores = ScenarioScores()(_inputs(window, forward), 1, samples, _mean(window))

        assert tuple(scores) == ScenarioScores.columns
        assert all(np.isfinite(value) for value in scores.values())
        pit = [scores[f"pit_bin_{i:02d}"] for i in range(PIT_BINS)]
        assert sum(pit) == pytest.approx(1.0)
        assert scores["ew_var_exceeded"] in (0.0, 1.0)
        assert 0.5 < scores["spread"] < 2.0

    def test_window_as_scenarios_matches_its_own_correlations(self, returns):
        window, forward = returns.iloc[:70], returns.iloc[70:]
        scores = ScenarioScores()(
            _inputs(window, forward), 1, window.to_numpy(), _mean(window)
        )
        assert scores["corr_error_trailing"] == 0.0
        assert scores["corr_bias_trailing"] == 0.0

    def test_scores_are_in_trailing_volatility_units(self, returns):
        window, forward = returns.iloc[:70], returns.iloc[70:]
        rng = np.random.default_rng(8)
        samples = rng.normal(0.0, 0.01, size=(400, 4))
        base = ScenarioScores()(_inputs(window, forward), 1, samples, _mean(window))
        scaled = ScenarioScores()(
            _inputs(10 * window, 10 * forward), 1, 10 * samples, _mean(window)
        )
        for column in ("crps", "energy_score", "spread"):
            assert scaled[column] == pytest.approx(base[column])

    def test_horizon_scores_the_h_day_outcome_on_the_root_h_scale(self, returns):
        """At horizon h the scores equal next-day scores of the h-day outcome
        against a window whose volatility is sqrt(h) times larger."""
        horizon = 4
        window, forward = returns.iloc[:70], returns.iloc[70:]
        rng = np.random.default_rng(12)
        samples = rng.multivariate_normal(
            np.zeros(4), horizon * window.cov().to_numpy(), size=500
        )
        outcome = ((1 + forward.iloc[:horizon]).prod() - 1).to_frame().T
        collapsed = pd.concat([outcome, forward.iloc[horizon:]])
        at_horizon = ScenarioScores()(
            _inputs(window, forward), horizon, samples, _mean(window)
        )
        next_day = ScenarioScores()(
            _inputs(np.sqrt(horizon) * window, collapsed), 1, samples, _mean(window)
        )
        for column in ScenarioScores.columns:
            # The forward correlations read every forward day, not the outcome.
            if column != "corr_error_forward":
                assert at_horizon[column] == pytest.approx(next_day[column]), column

    def test_horizon_beyond_the_forward_returns_is_refused(self, returns):
        """A horizon needs as many forward returns as it has days."""
        window, forward = returns.iloc[:70], returns.iloc[70:]
        with pytest.raises(ValueError, match="needs 11 forward returns"):
            ScenarioScores()(
                _inputs(window, forward), 11, window.to_numpy(), _mean(window)
            )

    def test_short_window_is_refused(self, returns):
        window = returns.iloc[: TRAILING_STD_WINDOW - 1]
        with pytest.raises(ValueError, match="trailing volatility"):
            ScenarioScores()(
                _inputs(window, returns.iloc[70:]),
                1,
                window.to_numpy(),
                _mean(window),
            )


def _factor_returns(rng, rows, assets, correlation):
    covariance = 0.0001 * ((1 - correlation) * np.eye(assets) + correlation)
    return rng.multivariate_normal(np.zeros(assets), covariance, size=rows)


class TestPortfolioScores:
    def test_returns_every_column(self, returns):
        window, forward = returns.iloc[:70], returns.iloc[70:]
        rng = np.random.default_rng(6)
        samples = rng.multivariate_normal(
            np.zeros(4), window.cov().to_numpy(), size=500
        )
        measurement = PortfolioScores((1, 2, 4), BOUNDS, 4)
        scores = measurement(_inputs(window, forward), 1, samples, _mean(window))

        assert tuple(scores) == measurement.columns
        assert all(np.isfinite(value) for value in scores.values())
        assert recorded_cells(measurement.columns) == list(measurement.cells)
        for kind, size in measurement.cells:
            pit = [
                scores[cell_column(kind, size, f"pit_bin_{i:02d}")]
                for i in range(PIT_BINS)
            ]
            assert sum(pit) == pytest.approx(1.0)

    def test_full_universe_equal_weights_reproduce_the_equal_weight_breach(
        self, returns
    ):
        window = returns.iloc[:70]
        rng = np.random.default_rng(9)
        measurement = PortfolioScores((4,), BOUNDS, 4)
        breaches = []
        for origin in range(40):
            samples = rng.normal(0.0, 0.006, size=(300, 4))
            # Outcomes wider than the scenarios, so some days breach.
            forward = pd.DataFrame(
                rng.normal(0.0, 0.012, size=(10, 4)),
                index=returns.index[70:],
                columns=returns.columns,
            )
            inputs = _inputs(window, forward, seed=origin)
            portfolio = measurement(inputs, 1, samples, _mean(window))
            asset = ScenarioScores()(inputs, 1, samples, _mean(window))
            assert (
                portfolio[cell_column("equal", 4, "breach_05")]
                == (asset["ew_var_exceeded"])
            )
            breaches.append(asset["ew_var_exceeded"])
        assert 0 < sum(breaches) < len(breaches)

    def test_crps_is_in_portfolio_volatility_units(self, returns):
        window, forward = returns.iloc[:70], returns.iloc[70:]
        rng = np.random.default_rng(8)
        samples = rng.normal(0.0, 0.0001, size=(400, 4))
        measurement = PortfolioScores((1, 4), BOUNDS, 4)
        base = measurement(_inputs(window, forward), 1, samples, _mean(window))
        scaled = measurement(
            _inputs(2 * window, 2 * forward), 1, 2 * samples, _mean(window)
        )
        assert scaled[PORTFOLIO_CRPS] == pytest.approx(base[PORTFOLIO_CRPS])

    def test_horizon_scores_hold_the_weights_over_the_h_day_outcome(self, returns):
        """At horizon h a portfolio's outcome is its buy-and-hold h-day return,
        and its CRPS is scaled by sqrt(h) times its trailing volatility."""
        horizon = 5
        window, forward = returns.iloc[:70], returns.iloc[70:]
        rng = np.random.default_rng(13)
        samples = rng.normal(0.0, 0.02, size=(400, 4))
        measurement = PortfolioScores((1, 2, 4), BOUNDS, 4)
        outcome = ((1 + forward.iloc[:horizon]).prod() - 1).to_frame().T
        at_horizon = measurement(
            _inputs(window, forward), horizon, samples, _mean(window)
        )
        next_day = measurement(
            _inputs(np.sqrt(horizon) * window, outcome), 1, samples, _mean(window)
        )
        for kind, size in measurement.cells:
            for reading in ("breach_05", "breach_01", "crps", "z2"):
                column = cell_column(kind, size, reading)
                assert at_horizon[column] == pytest.approx(next_day[column])

    def test_short_losing_more_than_the_capital_is_scored(self, returns):
        """A short position's loss has no floor, so a long-short portfolio can
        lose more than its capital in a scenario, and the scores still read it."""
        window, forward = returns.iloc[:70], returns.iloc[70:]
        rng = np.random.default_rng(14)
        samples = rng.normal(0.0, 0.01, size=(400, 4))
        # In row k asset k gains 5000 % while every other asset loses 90 %.
        samples[:4] = np.where(np.eye(4, dtype=bool), 50.0, -0.9)
        measurement = PortfolioScores((4,), BOUNDS, 4)
        inputs = _inputs(window, forward)
        weights = draw_portfolios(
            np.random.default_rng([inputs.seed, PORTFOLIO_STREAM]),
            4,
            measurement.cells,
            PORTFOLIO_SUBSETS,
            BOUNDS,
        )
        assert (weights @ samples.T).min() < -1
        scores = measurement(inputs, 1, samples, _mean(window))
        assert all(np.isfinite(value) for value in scores.values())

    def test_too_weak_correlation_breaches_more_as_portfolios_grow(self):
        rng = np.random.default_rng(10)
        assets, correlation = 10, 0.6
        dates = pd.date_range("2021-01-01", periods=80, freq="B")
        measurement = PortfolioScores((1, 10), BOUNDS, assets)
        readings = {"correct": [], "independent": []}
        for origin in range(300):
            history = pd.DataFrame(
                _factor_returns(rng, 80, assets, correlation), index=dates
            )
            window, forward = history.iloc[:70], history.iloc[70:]
            correct = _factor_returns(rng, 2000, assets, correlation)
            # Same marginals, no dependence.
            independent = rng.normal(0.0, 0.01, size=(2000, assets))
            inputs = _inputs(window, forward, seed=origin)
            for name, samples in (("correct", correct), ("independent", independent)):
                readings[name].append(measurement(inputs, 1, samples, _mean(window)))

        def mean(name, kind, size, reading):
            return np.mean(
                [row[cell_column(kind, size, reading)] for row in readings[name]]
            )

        assert mean("correct", "equal", 10, "breach_05") == pytest.approx(
            0.05, abs=0.03
        )
        assert mean("correct", "equal", 10, "z2") == pytest.approx(1.0, abs=0.25)
        assert mean("independent", "equal", 1, "breach_05") == pytest.approx(
            0.05, abs=0.03
        )
        # Scenario spread 0.4 of the real one: about a quarter of days breach.
        assert mean("independent", "equal", 10, "breach_05") > 0.18
        assert mean("independent", "equal", 10, "z2") > 3.0

    def test_size_above_the_universe_is_refused(self):
        with pytest.raises(ValueError, match="exceed"):
            PortfolioScores((1, 5), BOUNDS, 4)

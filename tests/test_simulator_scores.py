# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scores of the simulator comparison in experiments/ on synthetic data whose
answers are known."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist
from scipy.stats import norm

from experiments.simulator_comparison.scores import (
    PIT_BINS,
    TRAILING_STD_WINDOW,
    ScenarioScores,
    energy_score,
    interval_coverage,
    probability_integral_transform,
    sample_crps,
)
from experiments.simulator_comparison.trials import OriginInputs
from portfolio_optimization.cvar_data import CvarData


def _gaussian_crps(mean, std, realized):
    z = (realized - mean) / std
    return std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def _inputs(window, forward, seed=0):
    return OriginInputs(
        origin=window.index[-1],
        returns_dict={"returns": window},
        forward=forward,
        seed=seed,
    )


def _cvar_data(samples):
    count = samples.shape[0]
    return CvarData(
        mean=samples.mean(axis=0), R=samples.T, p=np.full(count, 1.0 / count)
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


class TestScenarioScores:
    def test_returns_every_column(self, returns):
        window, forward = returns.iloc[:70], returns.iloc[70:]
        rng = np.random.default_rng(6)
        samples = rng.multivariate_normal(
            np.zeros(4), window.cov().to_numpy(), size=500
        )
        scores = ScenarioScores()(_inputs(window, forward), _cvar_data(samples))

        assert tuple(scores) == ScenarioScores.columns
        assert all(np.isfinite(value) for value in scores.values())
        pit = [scores[f"pit_bin_{i:02d}"] for i in range(PIT_BINS)]
        assert sum(pit) == pytest.approx(1.0)
        assert scores["ew_var_exceeded"] in (0.0, 1.0)
        assert 0.5 < scores["spread"] < 2.0

    def test_window_as_scenarios_matches_its_own_correlations(self, returns):
        window, forward = returns.iloc[:70], returns.iloc[70:]
        scores = ScenarioScores()(
            _inputs(window, forward), _cvar_data(window.to_numpy())
        )
        assert scores["corr_error_trailing"] == 0.0
        assert scores["corr_bias_trailing"] == 0.0

    def test_scores_are_in_trailing_volatility_units(self, returns):
        window, forward = returns.iloc[:70], returns.iloc[70:]
        rng = np.random.default_rng(8)
        samples = rng.normal(0.0, 0.01, size=(400, 4))
        base = ScenarioScores()(_inputs(window, forward), _cvar_data(samples))
        scaled = ScenarioScores()(
            _inputs(10 * window, 10 * forward), _cvar_data(10 * samples)
        )
        for column in ("crps", "energy_score", "spread"):
            assert scaled[column] == pytest.approx(base[column])

    def test_short_window_is_refused(self, returns):
        window = returns.iloc[: TRAILING_STD_WINDOW - 1]
        with pytest.raises(ValueError, match="trailing volatility"):
            ScenarioScores()(
                _inputs(window, returns.iloc[70:]), _cvar_data(window.to_numpy())
            )

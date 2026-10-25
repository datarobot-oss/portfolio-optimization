# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Statistics and report of the simulator comparison in experiments/, on a
synthetic results table with planted answers."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from experiments.simulator_comparison.analysis import (
    PIT_COLUMNS,
    RELIABILITY_LEVELS,
    aggregate,
    comparisons,
    complete_trials,
    ranking,
    reliability,
    summary,
    timing,
)
from experiments.simulator_comparison.regimes import (
    FEATURES,
    OVERALL,
    REGIMES,
    check_prices,
    origin_features,
    regime_readings,
    terciles,
)
from experiments.simulator_comparison.report import write_report
from experiments.simulator_comparison.scores import (
    PIT_BINS,
    TRAILING_STD_WINDOW,
    ScenarioScores,
)
from experiments.simulator_comparison.table import RunMetadata, write_results
from experiments.simulator_comparison.trials import draw_trial

ARMS = {
    "no_fit-cpu": ("no_fit", "CPU"),
    "kde-cpu": ("kde", "CPU"),
    "kde-gpu": ("kde", "GPU"),
    "jointfm-gpu": ("jointfm", "GPU"),
}
TRIALS = 8
ORIGINS = 3
EXPERIMENT = {
    "seed": 0,
    "origins": ORIGINS,
    "assets": 5,
    "look_back": 70,
    "forward": 10,
}
PRICE_DAYS = 400
PRICE_COLUMNS = 8
# Planted: jointfm's energy score is this much below no_fit's in every trial.
ENERGY_SHIFT = -0.5
KDE_CPU_SECONDS = 0.08
KDE_GPU_SECONDS = 0.02
WARMUP_SECONDS = 100.0


def _prices():
    """Prices whose market loading and volatility drift, so origins differ in
    window correlation, volatility ratio, and correlation change."""
    rng = np.random.default_rng(1)
    time = np.arange(PRICE_DAYS)[:, None]
    loading = 1.0 + np.sin(time / 37.0)
    volatility = 0.01 * (1.5 + np.cos(time / 23.0))
    market = rng.normal(size=(PRICE_DAYS, 1))
    noise = rng.normal(size=(PRICE_DAYS, PRICE_COLUMNS))
    returns = volatility * (loading * market + noise)
    return pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=pd.bdate_range("2010-01-01", periods=PRICE_DAYS),
        columns=[f"A{i}" for i in range(PRICE_COLUMNS)],
    )


def _origin_dates(prices, trial):
    drawn = draw_trial(
        prices,
        trial,
        seed=EXPERIMENT["seed"],
        n_assets=EXPERIMENT["assets"],
        n_origins=ORIGINS,
        look_back=EXPERIMENT["look_back"],
        forward=EXPERIMENT["forward"],
    )
    return drawn, prices.index[list(drawn.origins)]


def _rows(prices):
    rng = np.random.default_rng(0)
    warmed = set()
    rows = []
    for trial in range(TRIALS):
        base = rng.normal(6.0, 0.3)
        for origin in _origin_dates(prices, trial)[1]:
            for arm, (fit_type, device) in ARMS.items():
                scores = {c: rng.uniform(0.0, 1.0) for c in ScenarioScores.columns}
                for column in PIT_COLUMNS:
                    scores[column] = 1 / PIT_BINS
                scores["energy_score"] = base + rng.normal(0.0, 0.01)
                if arm == "jointfm-gpu":
                    scores["energy_score"] += ENERGY_SHIFT
                seconds = {"kde-cpu": KDE_CPU_SECONDS, "kde-gpu": KDE_GPU_SECONDS}.get(
                    arm, 0.5
                )
                rows.append(
                    {
                        "trial": trial,
                        "origin": origin,
                        "arm": arm,
                        "fit_type": fit_type,
                        "device": device,
                        "assets_hash": f"h{trial}",
                        "seed": trial,
                        "num_scen": 100,
                        "seconds": WARMUP_SECONDS if arm not in warmed else seconds,
                        "warmup": arm not in warmed,
                        **scores,
                    }
                )
                warmed.add(arm)
    return pd.DataFrame(rows)


def _identity(prices_path, measured=True):
    return {
        "arms": {arm: {} for arm in ARMS},
        "prices": {
            "file": prices_path.name,
            "sha256": hashlib.sha256(prices_path.read_bytes()).hexdigest(),
        },
        "experiment": EXPERIMENT,
        "measurements": [{"columns": list(ScenarioScores.columns), "parameters": {}}]
        if measured
        else [],
    }


@pytest.fixture()
def prices():
    return _prices()


@pytest.fixture()
def prices_path(tmp_path, prices):
    path = tmp_path / "prices.csv"
    prices.to_csv(path)
    return path


@pytest.fixture()
def frame(prices):
    return _rows(prices)


@pytest.fixture()
def per_trial(frame):
    complete, _ = complete_trials(frame, tuple(ARMS), ORIGINS)
    return aggregate(complete)


class TestAnalysis:
    def test_one_row_per_trial_and_arm(self, per_trial):
        assert len(per_trial) == TRIALS * len(ARMS)
        assert not per_trial.index.duplicated().any()

    def test_planted_shift_is_recovered(self, per_trial):
        table = comparisons(per_trial)
        row = table[
            (table["metric"] == "energy_score") & (table["arm"] == "jointfm-gpu")
        ].iloc[0]
        assert row["reference"] == "no_fit-cpu"
        assert row["median_difference"] == pytest.approx(ENERGY_SHIFT, abs=0.02)
        assert row["ci_low"] <= row["median_difference"] <= row["ci_high"] < 0
        assert row["p_value"] < 0.05
        assert ranking(per_trial).index[0] == "jointfm-gpu"

    def test_gpu_is_compared_with_cpu_of_the_same_simulator(self, per_trial):
        table = comparisons(per_trial)
        pairs = set(zip(table["arm"], table["reference"]))
        assert ("kde-gpu", "kde-cpu") in pairs
        assert ("jointfm-gpu", "kde-cpu") not in pairs

    def test_warmup_calls_stay_out_of_the_timing(self, per_trial):
        means, speedups = timing(per_trial)
        assert means.loc["kde-gpu", "mean_seconds"] == pytest.approx(KDE_GPU_SECONDS)
        (speedup,) = speedups.itertuples()
        assert speedup.median_speedup == pytest.approx(
            KDE_CPU_SECONDS / KDE_GPU_SECONDS
        )

    def test_incomplete_trial_is_left_out(self, frame):
        incomplete = frame.drop(frame.index[-1])
        kept, excluded = complete_trials(incomplete, tuple(ARMS), ORIGINS)
        assert excluded == [TRIALS - 1]
        assert set(kept["trial"]) == set(range(TRIALS - 1))

    def test_flat_pit_is_reliable(self, per_trial):
        levels = reliability(per_trial)
        for arm in ARMS:
            np.testing.assert_allclose(levels.loc[arm], RELIABILITY_LEVELS)

    def test_missing_reference_is_refused(self, frame):
        without = frame[frame["arm"] != "no_fit-cpu"]
        complete, _ = complete_trials(
            without, tuple(a for a in ARMS if a != "no_fit-cpu"), ORIGINS
        )
        with pytest.raises(ValueError, match="no_fit-cpu"):
            comparisons(aggregate(complete))


class TestRegimes:
    def test_features_match_a_direct_computation(self, prices, frame):
        features = origin_features(prices, frame, EXPERIMENT)
        trial, dates = _origin_dates(prices, 0)
        position = trial.origins[0]
        log_prices = np.log(prices[list(trial.assets)].to_numpy())
        window = np.diff(
            log_prices[position - EXPERIMENT["look_back"] : position + 1], axis=0
        )
        forward = np.diff(
            log_prices[position : position + EXPERIMENT["forward"] + 1], axis=0
        )

        def mean_correlation(returns):
            matrix = np.corrcoef(returns.T)
            return matrix[np.triu_indices(len(matrix), 1)].mean()

        row = features.loc[(0, dates[0])]
        assert row["window_correlation"] == pytest.approx(mean_correlation(window))
        assert row["volatility_ratio"] == pytest.approx(
            np.median(window[-TRAILING_STD_WINDOW:].std(axis=0) / window.std(axis=0))
        )
        assert row["correlation_change"] == pytest.approx(
            mean_correlation(forward) - mean_correlation(window)
        )

    def test_terciles_split_origins_evenly_by_their_bounds(self, prices, frame):
        features = origin_features(prices, frame, EXPERIMENT)
        regimes, bounds = terciles(features)
        assert len(features) == TRIALS * ORIGINS
        for feature in FEATURES:
            rows = bounds[bounds["feature"] == feature].set_index("regime")
            assert list(rows["origins"]) == [TRIALS * ORIGINS // len(REGIMES)] * 3
            for name in REGIMES:
                members = features.loc[regimes[feature] == name, feature]
                assert members.between(
                    rows.loc[name, "low"], rows.loc[name, "high"]
                ).all()

    def test_planted_regime_effect_is_recovered(self, prices, frame):
        features = origin_features(prices, frame, EXPERIMENT)
        regimes, _ = terciles(features)
        high = regimes.index[regimes["window_correlation"] == "high"]
        at_high = pd.MultiIndex.from_frame(frame[["trial", "origin"]]).isin(high)
        frame["ew_var_exceeded"] = (at_high & (frame["arm"] == "jointfm-gpu")).astype(
            float
        )
        levels = regime_readings(frame, regimes)["summary"]
        breaches = levels[
            (levels["metric"] == "ew_var_exceeded") & (levels["arm"] == "jointfm-gpu")
        ].set_index(["feature", "regime"])["mean"]
        assert breaches[("window_correlation", "high")] == 1.0
        assert breaches[("window_correlation", "low")] == 0.0
        assert 0.0 < breaches[(OVERALL, OVERALL)] < 1.0

    def test_overall_readings_equal_the_overall_tables(self, prices, frame):
        regimes, _ = terciles(origin_features(prices, frame, EXPERIMENT))
        readings = regime_readings(frame, regimes)
        per_trial = aggregate(frame)
        overall = readings["comparisons"][readings["comparisons"]["feature"] == OVERALL]
        pd.testing.assert_frame_equal(
            overall.drop(columns=["feature", "regime"]).reset_index(drop=True),
            comparisons(per_trial),
        )
        levels = readings["summary"]
        expected = summary(per_trial)
        pd.testing.assert_frame_equal(
            levels[levels["feature"] == OVERALL]
            .drop(columns=["feature", "regime"])
            .reset_index(drop=True),
            expected[expected["metric"] != "seconds"].reset_index(drop=True),
        )

    def test_origin_not_drawn_from_the_prices_is_refused(self, prices, frame):
        frame["origin"] = frame["origin"] + pd.Timedelta(days=1)
        with pytest.raises(ValueError, match="does not draw"):
            origin_features(prices, frame, EXPERIMENT)

    def test_other_price_file_is_refused(self, tmp_path, prices_path):
        identity = _identity(prices_path)
        other = tmp_path / "other.csv"
        _prices().iloc[:-1].to_csv(other)
        with pytest.raises(ValueError, match="SHA-256"):
            check_prices(other, identity)


class TestReport:
    def test_every_file_is_written(self, tmp_path, prices_path, frame):
        results = tmp_path / "results.parquet"
        write_results(
            results, frame, RunMetadata(identity=_identity(prices_path), commands=[])
        )
        outputs = write_report(results, prices_path, tmp_path / "report")
        names = {path.name for path in outputs.files}
        plots = [
            "ranking",
            "scores",
            "skill",
            "calibration_pit",
            "calibration_reliability",
            "timing",
            *(f"regimes_{feature}" for feature in FEATURES),
        ]
        for plot in plots:
            assert {f"{plot}.png", f"{plot}.svg"} <= names
        tables = [
            "summary",
            "origin_features",
            "regimes",
            "regime_summary",
            "regime_comparisons",
            "regime_skill",
            "regime_ranking",
        ]
        assert {f"{table}.csv" for table in tables} <= names
        assert all(path.stat().st_size > 0 for path in outputs.files)
        assert outputs.excluded_trials == []

    def test_results_without_scores_are_refused(self, tmp_path, prices_path, frame):
        results = tmp_path / "results.parquet"
        write_results(
            results,
            frame,
            RunMetadata(identity=_identity(prices_path, measured=False), commands=[]),
        )
        with pytest.raises(ValueError, match="ScenarioScores"):
            write_report(results, prices_path, tmp_path / "report")

    def test_unfinished_arm_can_be_left_out(self, tmp_path, prices_path, frame):
        # One kde-gpu row in an extra trial: kde-gpu has not finished it.
        late = frame[(frame["trial"] == 0) & (frame["arm"] == "kde-gpu")].iloc[:1]
        results = tmp_path / "results.parquet"
        unfinished = pd.concat([frame, late.assign(trial=TRIALS)], ignore_index=True)
        write_results(
            results,
            unfinished,
            RunMetadata(identity=_identity(prices_path), commands=[]),
        )
        outputs = write_report(
            results, prices_path, tmp_path / "report", exclude_arms=["kde-gpu"]
        )
        assert outputs.excluded_trials == []
        ranking = pd.read_csv(tmp_path / "report" / "ranking.csv")
        assert "kde-gpu" not in set(ranking["arm"])
        assert pd.read_csv(tmp_path / "report" / "speedup.csv").empty

    def test_unknown_excluded_arm_is_refused(self, tmp_path, prices_path, frame):
        results = tmp_path / "results.parquet"
        write_results(
            results, frame, RunMetadata(identity=_identity(prices_path), commands=[])
        )
        with pytest.raises(ValueError, match="Cannot exclude"):
            write_report(
                results, prices_path, tmp_path / "report", exclude_arms=["jointfm-cpu"]
            )

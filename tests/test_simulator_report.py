# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Statistics and report of the simulator comparison in experiments/, on a
synthetic results table with planted answers."""

import hashlib
import itertools

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
from experiments.simulator_comparison.arms import (
    DERIVED_DONORS,
    INDEPENDENT_ARM,
    SWAP_DONORS,
)
from experiments.simulator_comparison.config import CvarConfig
from experiments.simulator_comparison.dilution import (
    dilution_lines,
    fit_line,
    origin_points,
)
from experiments.simulator_comparison.portfolio_analysis import (
    breach_band,
    portfolio_pit,
    size_curves,
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
    PORTFOLIO_CRPS,
    TRAILING_STD_WINDOW,
    PortfolioScores,
    ScenarioScores,
    cell_column,
)
from experiments.simulator_comparison.table import RunMetadata, write_results
from experiments.simulator_comparison.trials import draw_trial

ARMS = {
    "no_fit-cpu": ("no_fit", "CPU"),
    "kde-cpu": ("kde", "CPU"),
    "kde-gpu": ("kde", "GPU"),
    "jointfm-gpu": ("jointfm", "GPU"),
}
# The scenario comparison's measurement arms, with the identity entries that
# name their donors among ``ARMS``.
MEASUREMENT_ARMS = {
    "swap-jointfm-window": {"marginals": "jointfm-gpu", "dependence": "no_fit-cpu"},
    "swap-window-jointfm": {"marginals": "no_fit-cpu", "dependence": "jointfm-gpu"},
    "swap-window-window": {"marginals": "no_fit-cpu", "dependence": "no_fit-cpu"},
    INDEPENDENT_ARM: {"window": "no_fit-cpu"},
}
# The figures that read each measurement arm.
READERS = {
    **dict.fromkeys(
        SWAP_DONORS, ("swap_effects_horizons", "swap_effects_portfolio_sizes")
    ),
    INDEPENDENT_ARM: (
        "independence_control_horizons",
        "independence_control_portfolio_sizes",
    ),
}
TRIALS = 8
ORIGINS = 3
HORIZONS = (1, 5)
EXPERIMENT = {
    "seed": 0,
    "origins": ORIGINS,
    "assets": 5,
    "look_back": 70,
    "origin_history": 70,
    "forward": 10,
    "horizons": list(HORIZONS),
}
PRICE_DAYS = 400
PRICE_COLUMNS = 8
# Planted: jointfm's energy score is this much below no_fit's in every trial,
# times the horizon.
ENERGY_SHIFT = -0.5
# Planted: jointfm's portfolios are half as wide as their outcomes, and its
# portfolio CRPS is this share below no_fit's.
JOINTFM_Z2 = 4.0
JOINTFM_CRPS_SHARE = 0.8
PORTFOLIOS = PortfolioScores(
    (1, 2, 5),
    CvarConfig(
        w_min=-0.3,
        w_max=0.8,
        c_min=0.1,
        c_max=0.2,
        L_tar=1.6,
        T_tar=0.5,
        confidence=0.95,
        risk_aversion=1.0,
    ),
    EXPERIMENT["assets"],
)
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
        origin_history=EXPERIMENT["origin_history"],
        forward=EXPERIMENT["forward"],
    )
    return drawn, prices.index[list(drawn.origins)]


def _rows(prices, horizons, arms=ARMS):
    """Planted results of every trial, origin, arm of ``arms`` (name to fit
    type and device), and horizon of ``horizons``."""
    rng = np.random.default_rng(0)
    warmed = set()
    rows = []
    for trial in range(TRIALS):
        base = rng.normal(6.0, 0.3)
        for origin in _origin_dates(prices, trial)[1]:
            for arm, horizon in itertools.product(arms, horizons):
                fit_type, device = arms[arm]
                scores = {
                    c: rng.uniform(0.0, 1.0)
                    for c in (*ScenarioScores.columns, *PORTFOLIOS.columns)
                }
                for column in PIT_COLUMNS:
                    scores[column] = 1 / PIT_BINS
                for kind, size in PORTFOLIOS.cells:
                    for column in PIT_COLUMNS:
                        scores[cell_column(kind, size, column)] = 1 / PIT_BINS
                    jointfm = arm == "jointfm-gpu"
                    scores[cell_column(kind, size, "z2")] = (
                        JOINTFM_Z2 if jointfm else 1.0
                    )
                    scores[cell_column(kind, size, "crps")] = (
                        JOINTFM_CRPS_SHARE if jointfm else 1.0
                    )
                scores["energy_score"] = base + rng.normal(0.0, 0.01)
                if arm == "jointfm-gpu":
                    scores["energy_score"] += ENERGY_SHIFT * horizon
                seconds = {"kde-cpu": KDE_CPU_SECONDS, "kde-gpu": KDE_GPU_SECONDS}.get(
                    arm, 0.5
                )
                rows.append(
                    {
                        "trial": trial,
                        "origin": origin,
                        "arm": arm,
                        "horizon": horizon,
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
                if horizon == horizons[-1]:
                    warmed.add(arm)
    return pd.DataFrame(rows)


def _identity(prices_path, measured=2):
    """Identity of the planted table; ``measured`` counts the recorded measurements."""
    return {
        "arms": {arm: {} for arm in ARMS},
        "prices": {
            "file": prices_path.name,
            "sha256": hashlib.sha256(prices_path.read_bytes()).hexdigest(),
        },
        "experiment": EXPERIMENT,
        "measurements": [
            {"columns": list(ScenarioScores.columns), "parameters": {}},
            {"columns": list(PORTFOLIOS.columns), "parameters": {}},
        ][:measured],
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
    """Planted next-day rows, which every per-horizon statistic reads alike."""
    return _rows(prices, (1,))


@pytest.fixture()
def results(prices):
    """Planted rows of every horizon, as a results file holds them."""
    return _rows(prices, HORIZONS)


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    """The report of the planted rows of every horizon, rendered once per module."""
    directory = tmp_path_factory.mktemp("report")
    prices_path = directory / "prices.csv"
    prices = _prices()
    prices.to_csv(prices_path)
    path = directory / "results.parquet"
    write_results(
        path,
        _rows(prices, HORIZONS),
        RunMetadata(identity=_identity(prices_path), commands=[]),
    )
    return write_report(path, prices_path, directory / "report"), directory / "report"


@pytest.fixture()
def per_trial(frame):
    complete, _ = complete_trials(frame, tuple(ARMS), ORIGINS, 1)
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
        ranks = ranking(per_trial)
        energy = ranks[ranks["score"] == "energy_score"]
        assert energy["arm"].iloc[0] == "jointfm-gpu"
        assert set(ranks["score"]) == {
            "energy_score",
            "variogram_score",
            PORTFOLIO_CRPS,
        }

    def test_measurement_arms_are_not_ranked(self, per_trial):
        """A measurement arm that beats every simulator takes no win from one."""
        control = per_trial.xs("no_fit-cpu", level="arm", drop_level=False).rename(
            index={"no_fit-cpu": INDEPENDENT_ARM}, level="arm"
        )
        control["energy_score"] = -1.0
        ranks = ranking(pd.concat([per_trial, control]))
        assert set(ranks["arm"]) == set(ARMS)
        energy = ranks[ranks["score"] == "energy_score"].set_index("arm")
        assert energy.loc["jointfm-gpu", "share_won"] == 1.0

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
        kept, excluded = complete_trials(incomplete, tuple(ARMS), ORIGINS, 1)
        assert excluded == [TRIALS - 1]
        assert set(kept["trial"]) == set(range(TRIALS - 1))

    def test_flat_pit_is_reliable(self, per_trial):
        levels = reliability(per_trial)
        for arm in ARMS:
            np.testing.assert_allclose(levels.loc[arm], RELIABILITY_LEVELS)

    def test_planted_portfolio_readings_are_recovered(self, per_trial):
        curves = size_curves(per_trial).set_index(["kind", "size", "arm", "reading"])
        for kind, size in PORTFOLIOS.cells:
            ratio = curves.loc[(kind, size, "jointfm-gpu", "spread_ratio")]
            assert ratio["estimate"] == pytest.approx(1 / np.sqrt(JOINTFM_Z2))
            assert ratio["target"] == 1.0
            reference = curves.loc[(kind, size, "no_fit-cpu", "spread_ratio")]
            assert reference["estimate"] == pytest.approx(1.0)
            skill = curves.loc[(kind, size, "jointfm-gpu", "crps_skill")]
            assert skill["estimate"] == pytest.approx(100 * (1 - JOINTFM_CRPS_SHARE))
            breach = curves.loc[(kind, size, "kde-cpu", "breach_05")]
            band = breach_band(0.05, TRIALS * ORIGINS)
            assert (breach["band_low"], breach["band_high"]) == band
        assert ("long_short", 5) in PORTFOLIOS.cells
        assert ("long_short", 2) not in PORTFOLIOS.cells

    def test_breach_band_holds_the_level_and_narrows_with_origins(self):
        narrow, wide = breach_band(0.05, 4000), breach_band(0.05, 400)
        assert wide[0] < narrow[0] < 0.05 < narrow[1] < wide[1]

    def test_portfolio_pit_is_flat_where_planted_flat(self, per_trial):
        pit = portfolio_pit(per_trial)
        assert len(pit) == len(PORTFOLIOS.cells) * len(ARMS)
        np.testing.assert_allclose(pit[list(PIT_COLUMNS)], 1 / PIT_BINS)

    def test_rows_of_several_horizons_are_not_mixed(self, results):
        """aggregate refuses rows whose horizons would mix in one mean."""
        with pytest.raises(ValueError, match="one horizon"):
            aggregate(results)

    def test_trial_missing_a_horizon_is_incomplete(self, results):
        """A trial is complete only with every origin at every horizon."""
        incomplete = results.drop(results.index[-1])
        _, excluded = complete_trials(incomplete, tuple(ARMS), ORIGINS, len(HORIZONS))
        assert excluded == [TRIALS - 1]

    def test_missing_reference_is_refused(self, frame):
        without = frame[frame["arm"] != "no_fit-cpu"]
        complete, _ = complete_trials(
            without, tuple(a for a in ARMS if a != "no_fit-cpu"), ORIGINS, 1
        )
        with pytest.raises(ValueError, match="no_fit-cpu"):
            comparisons(aggregate(complete))


class TestRegimes:
    def test_features_match_a_direct_computation(self, prices, frame):
        features = origin_features(prices, frame, EXPERIMENT)
        trial, dates = _origin_dates(prices, 0)
        position = trial.origins[0]
        universe = prices[list(trial.assets)].to_numpy()

        def simple_returns(rows):
            return rows[1:] / rows[:-1] - 1

        window = simple_returns(
            universe[position - EXPERIMENT["look_back"] : position + 1]
        )
        forward = simple_returns(
            universe[position : position + EXPERIMENT["forward"] + 1]
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


# Planted dilution: jointfm's bias is this line of the window correlation.
DILUTION_INTERCEPT = -0.036
DILUTION_SLOPE = -0.48
DILUTION_NOISE = 0.01
DILUTION_TRIALS = 40


@pytest.fixture()
def dilution_rows():
    """Next-day bias rows on the planted line, and their origins' features."""
    rng = np.random.default_rng(2)
    origins = pd.bdate_range("2015-01-01", periods=ORIGINS)
    keys = pd.MultiIndex.from_product(
        [range(DILUTION_TRIALS), origins], names=["trial", "origin"]
    )
    window = rng.uniform(0.1, 0.6, size=len(keys))
    features = pd.DataFrame(
        {
            "window_correlation": window,
            "volatility_ratio": rng.uniform(0.7, 1.4, size=len(keys)),
            "correlation_change": rng.normal(0.0, 0.1, size=len(keys)),
        },
        index=keys,
    )
    bias = {
        "no_fit-cpu": np.zeros(len(keys)),
        "jointfm-gpu": DILUTION_INTERCEPT
        + DILUTION_SLOPE * window
        + rng.normal(0.0, DILUTION_NOISE, size=len(keys)),
    }
    frame = pd.concat(
        [
            keys.to_frame(index=False).assign(
                arm=arm, horizon=1, corr_bias_trailing=values
            )
            for arm, values in bias.items()
        ],
        ignore_index=True,
    )
    return frame, features


class TestDilution:
    def test_planted_line_is_recovered(self, dilution_rows):
        frame, features = dilution_rows
        regimes, _ = terciles(features)
        lines = dilution_lines(frame, features, regimes).set_index(
            ["feature", "regime", "arm"]
        )
        jointfm = lines.loc[(OVERALL, OVERALL, "jointfm-gpu")]
        assert jointfm["intercept"] == pytest.approx(DILUTION_INTERCEPT, abs=0.01)
        assert jointfm["slope"] == pytest.approx(DILUTION_SLOPE, abs=0.02)
        assert jointfm["forecast_slope"] == pytest.approx(1 + jointfm["slope"])
        assert jointfm["slope_low"] < DILUTION_SLOPE < jointfm["slope_high"]
        assert jointfm["intercept_low"] < DILUTION_INTERCEPT < jointfm["intercept_high"]
        assert jointfm["origins"] == DILUTION_TRIALS * ORIGINS
        window = lines.loc[(OVERALL, OVERALL, "no_fit-cpu")]
        assert window["intercept"] == pytest.approx(0.0, abs=1e-12)
        assert window["slope"] == pytest.approx(0.0, abs=1e-12)

    def test_fit_equals_least_squares_over_all_origins(self, dilution_rows):
        frame, features = dilution_rows
        points = origin_points(frame, features)
        cell = points[points["arm"] == "jointfm-gpu"]
        (intercept, slope), resamples = fit_line(cell, "jointfm-gpu")
        expected_slope, expected_intercept = np.polyfit(
            cell["window_correlation"], cell["corr_bias_trailing"], 1
        )
        assert slope == pytest.approx(expected_slope)
        assert intercept == pytest.approx(expected_intercept)
        assert resamples.shape[1] == 2

    def test_every_regime_holds_its_origins(self, dilution_rows):
        frame, features = dilution_rows
        regimes, _ = terciles(features)
        lines = dilution_lines(frame, features, regimes)
        assert len(lines) == (1 + len(FEATURES) * len(REGIMES)) * 2
        per_feature = lines[lines["arm"] == "jointfm-gpu"].groupby("feature")
        assert (per_feature["origins"].sum() == DILUTION_TRIALS * ORIGINS).all()

    def test_one_window_correlation_is_refused(self, dilution_rows):
        frame, features = dilution_rows
        features["window_correlation"] = 0.3
        with pytest.raises(ValueError, match="no line"):
            fit_line(origin_points(frame, features), "every arm")

    def test_resample_without_a_line_leaves_the_interval_undefined(self, dilution_rows):
        """Each regime holds one origin of each of two trials, so half its
        resamples draw one origin twice; all origins together keep a line."""
        frame, features = dilution_rows
        two = frame[frame["trial"] < 2]
        position = features.groupby("trial").cumcount().to_numpy()
        regimes = pd.DataFrame(
            {feature: np.array(REGIMES)[position] for feature in FEATURES},
            index=features.index,
        )
        lines = dilution_lines(two, features, regimes)
        jointfm = lines[lines["arm"] == "jointfm-gpu"].set_index(["feature", "regime"])
        bounds = ["slope_low", "slope_high", "intercept_low", "intercept_high"]
        overall = jointfm.loc[(OVERALL, OVERALL)]
        assert overall["undefined_resamples"] == 0
        assert np.isfinite(overall[bounds].to_numpy(float)).all()
        thin = jointfm.drop(index=(OVERALL, OVERALL))
        assert (thin["undefined_resamples"] > 0).all()
        assert np.isfinite(thin["slope"]).all()
        assert thin[bounds].isna().all().all()

    def test_rows_of_two_horizons_are_refused(self, dilution_rows):
        frame, features = dilution_rows
        both = pd.concat([frame, frame.assign(horizon=5)], ignore_index=True)
        with pytest.raises(ValueError, match="one horizon"):
            origin_points(both, features)


class TestReport:
    def test_every_file_is_written(self, report):
        outputs, directory = report
        names = {written.name for written in outputs.files}
        plots = [
            "ranking",
            "scores",
            "skill",
            "calibration_pit",
            "calibration_reliability",
            "timing",
            *(f"regimes_{feature}" for feature in FEATURES),
            "dilution",
            "portfolio_sizes",
            "portfolio_pit_equal",
            "portfolio_pit_long_only",
            "portfolio_pit_long_short",
            "horizons",
            "portfolio_horizons_equal",
            "portfolio_horizons_long_only",
            "portfolio_horizons_long_short",
        ]
        for plot in plots:
            assert {f"{plot}.png", f"{plot}.svg"} <= names
        # Without the swap arms there are no effects to draw.
        assert not any("swap_effects" in name for name in names)
        tables = [
            "summary",
            "origin_features",
            "regimes",
            "regime_summary",
            "regime_comparisons",
            "regime_skill",
            "regime_ranking",
            "regime_portfolio_sizes",
            "dilution",
            "portfolio_sizes",
            "portfolio_pit",
        ]
        assert {f"{table}.csv" for table in tables} <= names
        assert all(written.stat().st_size > 0 for written in outputs.files)
        assert outputs.excluded_trials == []
        for table in set(tables) - {"origin_features", "regimes"}:
            stacked = pd.read_csv(directory / f"{table}.csv")
            assert sorted(stacked["horizon"].unique()) == list(HORIZONS), table

    def test_planted_horizon_effect_is_recovered(self, report):
        """Each horizon's readings come from that horizon's rows alone."""
        _, directory = report
        table = pd.read_csv(directory / "comparisons.csv")
        energy = table[
            (table["metric"] == "energy_score")
            & (table["arm"] == "jointfm-gpu")
            & (table["reference"] == "no_fit-cpu")
        ].set_index("horizon")["median_difference"]
        for horizon in HORIZONS:
            assert energy[horizon] == pytest.approx(ENERGY_SHIFT * horizon, abs=0.02)
        timing_table = pd.read_csv(directory / "timing.csv")
        kde_gpu = timing_table.set_index("arm").loc["kde-gpu", "mean_seconds"]
        assert kde_gpu == pytest.approx(KDE_GPU_SECONDS)

    def test_results_without_horizons_are_refused(self, tmp_path, prices_path, frame):
        """A results file recorded before horizons existed cannot be reported."""
        path = tmp_path / "results.parquet"
        identity = _identity(prices_path)
        identity["experiment"] = {
            k: v for k, v in EXPERIMENT.items() if k != "horizons"
        }
        write_results(path, frame, RunMetadata(identity=identity, commands=[]))
        with pytest.raises(ValueError, match="no horizons"):
            write_report(path, prices_path, tmp_path / "report")

    @pytest.mark.parametrize(
        "measured, missing", [(0, "ScenarioScores"), (1, "PortfolioScores")]
    )
    def test_results_without_scores_are_refused(
        self, tmp_path, prices_path, frame, measured, missing
    ):
        results = tmp_path / "results.parquet"
        write_results(
            results,
            frame,
            RunMetadata(
                identity=_identity(prices_path, measured=measured), commands=[]
            ),
        )
        with pytest.raises(ValueError, match=missing):
            write_report(results, prices_path, tmp_path / "report")

    def test_unfinished_arm_can_be_left_out(self, tmp_path, prices_path, results):
        # One kde-gpu row in an extra trial: kde-gpu has not finished it.
        late = results[(results["trial"] == 0) & (results["arm"] == "kde-gpu")]
        path = tmp_path / "results.parquet"
        unfinished = pd.concat(
            [results, late.iloc[:1].assign(trial=TRIALS)], ignore_index=True
        )
        write_results(
            path,
            unfinished,
            RunMetadata(identity=_identity(prices_path), commands=[]),
        )
        outputs = write_report(
            path, prices_path, tmp_path / "report", exclude_arms=["kde-gpu"]
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


def test_measurement_arms_are_drawn_only_by_their_readers(
    tmp_path, prices, prices_path, saved_figures
):
    """Every per-arm figure names the simulators alone, and every measurement
    arm's reader is drawn."""
    arms = {**ARMS, **{arm: (arm, "CPU") for arm in MEASUREMENT_ARMS}}
    identity = _identity(prices_path)
    identity["arms"].update(MEASUREMENT_ARMS)
    path = tmp_path / "results.parquet"
    write_results(
        path, _rows(prices, HORIZONS, arms), RunMetadata(identity=identity, commands=[])
    )
    write_report(path, prices_path, tmp_path / "report")

    readers = {name for names in READERS.values() for name in names}
    per_arm = set(saved_figures) - readers
    assert {"ranking", "scores", "horizons", "portfolio_sizes"} <= per_arm
    for name in per_arm:
        assert "jointfm" in saved_figures[name], name
        named = [arm for arm in DERIVED_DONORS if arm in saved_figures[name]]
        assert named == [], name
    for arm, names in READERS.items():
        assert set(names) <= set(saved_figures), arm
    for name in READERS[INDEPENDENT_ARM]:
        assert INDEPENDENT_ARM in saved_figures[name]
    control = pd.read_csv(tmp_path / "report" / "independence_control.csv")
    assert set(control["horizon"]) == set(HORIZONS)
    assert "portfolio_equal_5_breach_05" in set(control["metric"])

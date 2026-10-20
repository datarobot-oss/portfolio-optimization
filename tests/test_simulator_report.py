# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Statistics and report of the simulator comparison in experiments/, on a
synthetic results table with planted answers."""

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
    timing,
)
from experiments.simulator_comparison.report import write_report
from experiments.simulator_comparison.scores import PIT_BINS, ScenarioScores
from experiments.simulator_comparison.table import RunMetadata, write_results

ARMS = {
    "no_fit-cpu": ("no_fit", "CPU"),
    "kde-cpu": ("kde", "CPU"),
    "kde-gpu": ("kde", "GPU"),
    "jointfm-gpu": ("jointfm", "GPU"),
}
TRIALS = 8
ORIGINS = 3
# Planted: jointfm's energy score is this much below no_fit's in every trial.
ENERGY_SHIFT = -0.5
KDE_CPU_SECONDS = 0.08
KDE_GPU_SECONDS = 0.02
WARMUP_SECONDS = 100.0


def _rows():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-01", periods=ORIGINS, freq="MS")
    warmed = set()
    rows = []
    for trial in range(TRIALS):
        base = rng.normal(6.0, 0.3)
        for origin in dates:
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


def _identity(measured=True):
    return {
        "arms": {arm: {} for arm in ARMS},
        "experiment": {"origins": ORIGINS},
        "measurements": [{"columns": list(ScenarioScores.columns), "parameters": {}}]
        if measured
        else [],
    }


@pytest.fixture()
def frame():
    return _rows()


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


class TestReport:
    def test_every_file_is_written(self, tmp_path, frame):
        results = tmp_path / "results.parquet"
        write_results(results, frame, RunMetadata(identity=_identity(), commands=[]))
        outputs = write_report(results, tmp_path / "report")
        names = {path.name for path in outputs.files}
        for plot in (
            "ranking",
            "scores",
            "skill",
            "calibration_pit",
            "calibration_reliability",
            "timing",
        ):
            assert {f"{plot}.png", f"{plot}.svg"} <= names
        assert "summary.csv" in names
        assert all(path.stat().st_size > 0 for path in outputs.files)
        assert outputs.excluded_trials == []

    def test_results_without_scores_are_refused(self, tmp_path, frame):
        results = tmp_path / "results.parquet"
        write_results(
            results, frame, RunMetadata(identity=_identity(measured=False), commands=[])
        )
        with pytest.raises(ValueError, match="ScenarioScores"):
            write_report(results, tmp_path / "report")

    def test_unfinished_arm_can_be_left_out(self, tmp_path, frame):
        # One kde-gpu row in an extra trial: kde-gpu has not finished it.
        late = frame[(frame["trial"] == 0) & (frame["arm"] == "kde-gpu")].iloc[:1]
        results = tmp_path / "results.parquet"
        unfinished = pd.concat([frame, late.assign(trial=TRIALS)], ignore_index=True)
        write_results(
            results, unfinished, RunMetadata(identity=_identity(), commands=[])
        )
        outputs = write_report(results, tmp_path / "report", exclude_arms=["kde-gpu"])
        assert outputs.excluded_trials == []
        ranking = pd.read_csv(tmp_path / "report" / "ranking.csv")
        assert "kde-gpu" not in set(ranking["arm"])
        assert pd.read_csv(tmp_path / "report" / "speedup.csv").empty

    def test_unknown_excluded_arm_is_refused(self, tmp_path, frame):
        results = tmp_path / "results.parquet"
        write_results(results, frame, RunMetadata(identity=_identity(), commands=[]))
        with pytest.raises(ValueError, match="Cannot exclude"):
            write_report(results, tmp_path / "report", exclude_arms=["jointfm-cpu"])

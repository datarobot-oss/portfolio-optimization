# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the origin history and the history comparison in experiments/,
with the fake JointFM client of conftest.py."""

import copy

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from experiments.simulator_comparison.arms import ARMS_BY_NAME
from experiments.simulator_comparison.config import ExperimentConfig
from experiments.simulator_comparison.history import history_comparison
from experiments.simulator_comparison.runner import pending_work, run_experiment
from experiments.simulator_comparison.scores import PortfolioScores, ScenarioScores
from experiments.simulator_comparison.table import (
    RunMetadata,
    read_results,
    write_results,
)
from experiments.simulator_comparison.trials import draw_trial, origin_inputs

# The scores need 63 returns for the trailing volatility.
SHORT_LOOK_BACK = 64
LONG_LOOK_BACK = 80
FORWARD = 8
ROWS = 120
CONFIG_ARMS = ("no_fit-cpu", "jointfm")
RUN_ARMS = ("no_fit-cpu", "jointfm-cpu")
# Planted per-trial change of the JointFM arm's CRPS in a copied results file.
PLANTED_CRPS_CHANGE = 0.25


@pytest.fixture()
def prices():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=ROWS, freq="B")
    log_prices = np.cumsum(rng.normal(0.0, 0.01, size=(ROWS, 6)), axis=0)
    return pd.DataFrame(
        100.0 * np.exp(log_prices), index=dates, columns=[f"A{i}" for i in range(6)]
    )


@pytest.fixture()
def prices_path(tmp_path, prices):
    path = tmp_path / "prices.csv"
    prices.to_csv(path)
    return path


def _config(tmp_path, prices_path, look_back, **overrides):
    values = dict(
        prices=prices_path,
        output=tmp_path / f"results_{look_back}.parquet",
        trials=2,
        origins=3,
        assets=4,
        look_back=look_back,
        origin_history=LONG_LOOK_BACK,
        forward=FORWARD,
        num_scen=16,
        seed=5,
        arms=CONFIG_ARMS,
        portfolio_sizes=(1, 4),
        horizons=(1,),
        rebalancing=dict(
            output=tmp_path / f"rebalancing_{look_back}.parquet",
            arms=("no_fit-cpu",),
            trials=2,
            trading_days=10,
            look_forward_window=5,
            look_back_days=90,
            trigger="every_step",
            pct_change_threshold=-0.005,
            transaction_cost=0.001,
            risk_free_rates=tmp_path / "rates.csv",
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
        ),
    )
    values.update(overrides)
    return ExperimentConfig(**values)


def _run(config):
    return run_experiment(
        config,
        [
            ScenarioScores(),
            PortfolioScores(
                config.portfolio_sizes, config.rebalancing.cvar, config.assets
            ),
        ],
    )


@pytest.fixture()
def runs(tmp_path, prices_path, fake_jointfm):
    """The short- and long-history configs, both run on the same fake service."""
    fake_jointfm.envelope.n_input = 1000
    short = _config(tmp_path, prices_path, SHORT_LOOK_BACK)
    long = _config(tmp_path, prices_path, LONG_LOOK_BACK)
    _run(short)
    _run(long)
    return short, long


def _copy_results(source, target, edit_identity, edit_frame=None):
    """Write ``source``'s rows to ``target`` with an edited identity and frame."""
    frame, metadata = read_results(source)
    identity = copy.deepcopy(metadata.identity)
    edit_identity(identity)
    if edit_frame is not None:
        edit_frame(frame)
    write_results(target, frame, RunMetadata(identity, metadata.commands))
    return target


def _same_look_back(identity):
    """Leave the identity as it is, so both files hold one look_back."""


def _other_num_scen(identity):
    """Another look_back and, beside it, another scenario count."""
    identity["experiment"].update(look_back=1, num_scen=8)


def _other_origins(identity):
    """Another look_back and, beside it, origins drawn from another history."""
    identity["experiment"].update(look_back=1, origin_history=1)


def _other_checkpoint(identity):
    """Another look_back and, beside it, a JointFM service of another checkpoint."""
    identity["experiment"]["look_back"] = 1
    for service in identity["services"].values():
        service["checkpoint_version"] = "other-checkpoint"


class TestOriginHistory:
    def test_origins_do_not_depend_on_look_back(self, tmp_path, prices, prices_path):
        arms = [ARMS_BY_NAME["no_fit-cpu"]]
        drawn = [
            [
                (trial, origin)
                for trial, origin, _ in pending_work(
                    prices, _config(tmp_path, prices_path, look_back), arms, set()
                )
            ]
            for look_back in (SHORT_LOOK_BACK, LONG_LOOK_BACK)
        ]
        assert drawn[0] == drawn[1]
        assert min(o for _, o in drawn[0]) >= LONG_LOOK_BACK

    def test_every_window_holds_look_back_returns(self, prices):
        trial = draw_trial(
            prices,
            0,
            seed=5,
            n_assets=4,
            n_origins=1,
            origin_history=LONG_LOOK_BACK,
            forward=FORWARD,
        )
        (origin,) = trial.origins
        windows = {
            look_back: origin_inputs(
                prices, trial, origin, seed=5, look_back=look_back, forward=FORWARD
            ).returns_dict["returns"]
            for look_back in (SHORT_LOOK_BACK, LONG_LOOK_BACK)
        }
        assert {k: len(w) for k, w in windows.items()} == {
            SHORT_LOOK_BACK: SHORT_LOOK_BACK,
            LONG_LOOK_BACK: LONG_LOOK_BACK,
        }
        pd.testing.assert_frame_equal(
            windows[SHORT_LOOK_BACK], windows[LONG_LOOK_BACK].iloc[-SHORT_LOOK_BACK:]
        )

    def test_origin_history_below_look_back_is_refused(self, tmp_path, prices_path):
        with pytest.raises(ValidationError, match="must be at least look_back"):
            _config(tmp_path, prices_path, SHORT_LOOK_BACK, origin_history=63)


class TestHistoryComparison:
    def test_runs_pair_by_trial_and_jointfm_receives_its_history(
        self, runs, fake_jointfm
    ):
        short, long = runs
        keys = [
            read_results(config.output)[0][["trial", "origin", "seed"]]
            .drop_duplicates()
            .reset_index(drop=True)
            for config in runs
        ]
        pd.testing.assert_frame_equal(keys[0], keys[1])
        # A window of n returns is sent as its n + 1 log prices.
        lengths = sorted({len(call["history"]) for call in fake_jointfm.forecast_calls})
        assert lengths == [SHORT_LOOK_BACK + 1, LONG_LOOK_BACK + 1]

        table = history_comparison(short.output, long.output)
        assert set(table["arm"]) == set(RUN_ARMS)
        assert (table["trials"] == short.trials).all()
        assert (table["look_back"] == LONG_LOOK_BACK).all()
        assert (table["baseline_look_back"] == SHORT_LOOK_BACK).all()
        # no_fit's scenarios are its window, which reads zero bias at any length.
        no_fit_bias = table[
            (table["arm"] == "no_fit-cpu") & (table["metric"] == "corr_bias_trailing")
        ]
        assert no_fit_bias["median_difference"].item() == 0.0

    def test_a_planted_change_is_recovered(self, runs, tmp_path):
        short, _ = runs

        def other_look_back(identity):
            identity["experiment"]["look_back"] = LONG_LOOK_BACK

        def worse_jointfm_crps(frame):
            frame.loc[frame["arm"] == "jointfm-cpu", "crps"] += PLANTED_CRPS_CHANGE

        other = _copy_results(
            short.output,
            tmp_path / "planted.parquet",
            other_look_back,
            worse_jointfm_crps,
        )
        table = history_comparison(short.output, other).set_index(["arm", "metric"])
        jointfm = table.loc[("jointfm-cpu", "crps")]
        assert jointfm["median_difference"] == pytest.approx(PLANTED_CRPS_CHANGE)
        assert jointfm["ci_low"] == pytest.approx(PLANTED_CRPS_CHANGE)
        assert jointfm["ci_high"] == pytest.approx(PLANTED_CRPS_CHANGE)
        assert table.loc[("no_fit-cpu", "crps"), "median_difference"] == 0.0
        assert table.loc[("jointfm-cpu", "energy_score"), "median_difference"] == 0.0

    def test_another_host_is_accepted(self, runs, tmp_path):
        short, long = runs

        def other_host(identity):
            identity["host"]["hostname"] = "another-host"

        other = _copy_results(long.output, tmp_path / "host.parquet", other_host)
        assert len(history_comparison(short.output, other))

    @pytest.mark.parametrize(
        "edit, message",
        [
            (_same_look_back, "both hold look_back"),
            (_other_num_scen, "'num_scen': 8"),
            (_other_origins, "'origin_history': 1"),
            (_other_checkpoint, "services.jointfm-cpu"),
        ],
        ids=["same look_back", "other num_scen", "other origins", "other checkpoint"],
    )
    def test_other_conditions_are_refused(self, runs, tmp_path, edit, message):
        short, _ = runs
        other = _copy_results(short.output, tmp_path / "other.parquet", edit)
        with pytest.raises(ValueError, match=message):
            history_comparison(short.output, other)

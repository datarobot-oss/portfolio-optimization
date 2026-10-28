# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the simulator-comparison runner in experiments/, with the fake
JointFM client of conftest.py."""

import numpy as np
import pandas as pd
import pytest

from experiments.simulator_comparison.config import ExperimentConfig
from experiments.simulator_comparison.runner import run_experiment
from experiments.simulator_comparison.scores import PortfolioScores, ScenarioScores
from experiments.simulator_comparison.table import read_results
from experiments.simulator_comparison.trials import draw_trial, origin_inputs

# Configured arms; the fake service runs on CPU, so jointfm runs as jointfm-cpu.
CONFIG_ARMS = ("no_fit-cpu", "gaussian-cpu", "kde-cpu", "jointfm")
CPU_ARMS = ("no_fit-cpu", "gaussian-cpu", "kde-cpu", "jointfm-cpu")
# The scores need 63 returns for the trailing volatility.
LOOK_BACK = 64
FORWARD = 8
N_ORIGINS = 3
ROWS = 120


@pytest.fixture()
def prices():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=ROWS, freq="B")
    log_prices = np.cumsum(rng.normal(0.0, 0.01, size=(ROWS, 8)), axis=0)
    return pd.DataFrame(
        100.0 * np.exp(log_prices), index=dates, columns=[f"A{i}" for i in range(8)]
    )


@pytest.fixture()
def prices_path(tmp_path, prices):
    path = tmp_path / "prices.csv"
    prices.to_csv(path)
    return path


def _config(tmp_path, prices_path, **overrides):
    values = dict(
        prices=prices_path,
        output=tmp_path / "results.parquet",
        trials=2,
        origins=N_ORIGINS,
        assets=4,
        look_back=LOOK_BACK,
        origin_history=LOOK_BACK,
        forward=FORWARD,
        num_scen=16,
        seed=5,
        arms=CONFIG_ARMS,
        portfolio_sizes=(1, 2, 4),
        horizons=(1,),
        rebalancing=dict(
            output=tmp_path / "rebalancing.parquet",
            scenario_results=tmp_path / "results.parquet",
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
        universe_sizes=dict(
            output=tmp_path / "universe_sizes.parquet",
            arms=("no_fit-cpu",),
            sizes=(4,),
        ),
    )
    values.update(overrides)
    return ExperimentConfig(**values)


class RecordingMeasurement:
    """Records what each call received and returns one value."""

    columns = ("scenario_rows",)
    parameters = {}

    def __init__(self):
        self.calls = []

    def __call__(self, inputs, horizon, samples, expected_returns):
        self.calls.append((inputs, horizon, samples))
        return {"scenario_rows": float(samples.shape[1])}


class TestTrials:
    def test_trial_is_reproducible_from_its_index(self, prices):
        kwargs = dict(
            seed=5,
            n_assets=4,
            n_origins=N_ORIGINS,
            origin_history=LOOK_BACK,
            forward=FORWARD,
        )
        trial = draw_trial(prices, 1, **kwargs)
        assert trial == draw_trial(prices, 1, **kwargs)
        assert trial != draw_trial(prices, 2, **kwargs)
        assert len(set(trial.assets)) == 4
        assert list(trial.assets) == [c for c in prices.columns if c in trial.assets]
        assert list(trial.origins) == sorted(set(trial.origins))
        assert min(trial.origins) >= LOOK_BACK
        assert max(trial.origins) <= len(prices) - 1 - FORWARD

    def test_forward_returns_follow_the_origin(self, prices):
        trial = draw_trial(
            prices,
            0,
            seed=5,
            n_assets=4,
            n_origins=1,
            origin_history=LOOK_BACK,
            forward=FORWARD,
        )
        (origin,) = trial.origins
        inputs = origin_inputs(
            prices, trial, origin, seed=5, look_back=LOOK_BACK, forward=FORWARD
        )
        universe = prices[list(trial.assets)]
        expected = universe.pct_change().iloc[origin + 1 : origin + 1 + FORWARD]
        pd.testing.assert_frame_equal(inputs.forward, expected)
        pd.testing.assert_series_equal(
            inputs.realized(1), expected.iloc[0], check_names=False
        )
        # The h-day outcome compounds the daily returns: the price ratio minus one.
        pd.testing.assert_series_equal(
            inputs.realized(FORWARD),
            universe.iloc[origin + FORWARD] / universe.iloc[origin] - 1,
            check_names=False,
        )
        with pytest.raises(ValueError, match="needs 9 forward returns"):
            inputs.realized(FORWARD + 1)
        assert inputs.returns_dict["returns"].index[-1] == prices.index[origin]

    def test_too_many_assets_are_refused(self, prices):
        with pytest.raises(ValueError, match="draws 9 assets"):
            draw_trial(
                prices,
                0,
                seed=5,
                n_assets=9,
                n_origins=1,
                origin_history=LOOK_BACK,
                forward=FORWARD,
            )


class TestRunner:
    def test_arms_receive_identical_inputs(self, tmp_path, prices_path, fake_jointfm):
        fake_jointfm.envelope.n_input = 1000
        measurement = RecordingMeasurement()
        frame = run_experiment(_config(tmp_path, prices_path), [measurement])

        assert len(frame) == 2 * N_ORIGINS * len(CPU_ARMS)
        per_origin = frame.groupby(["trial", "origin"])
        assert (per_origin["arm"].nunique() == len(CPU_ARMS)).all()
        assert (per_origin["seed"].nunique() == 1).all()
        assert (frame.groupby("trial")["assets_hash"].nunique() == 1).all()
        assert (frame["scenario_rows"] == 4).all()

        for inputs, horizon, samples in measurement.calls:
            # No arm sees the cvar_data another arm added to its copy.
            assert "cvar_data" not in inputs.returns_dict
            assert horizon == 1
        no_fit_calls = measurement.calls[:: len(CPU_ARMS)]
        jointfm_rows = frame[frame["arm"] == "jointfm-cpu"]
        assert [call["seed"] for call in fake_jointfm.forecast_calls] == list(
            jointfm_rows["seed"]
        )
        for (inputs, _, samples), call in zip(
            no_fit_calls, fake_jointfm.forecast_calls
        ):
            window = inputs.returns_dict["returns"]
            # no_fit returns the window itself, and JointFM got its log-price path.
            np.testing.assert_array_equal(samples, window.to_numpy())
            np.testing.assert_allclose(
                call["history"].to_numpy()[1:],
                np.cumsum(np.log1p(window.to_numpy()), axis=0),
            )
        assert (frame.loc[frame["arm"] == "no_fit-cpu", "num_scen"] == LOOK_BACK).all()
        assert (frame.loc[frame["arm"] != "no_fit-cpu", "num_scen"] == 16).all()

    def test_resumption_adds_no_duplicate_rows(
        self, tmp_path, prices_path, fake_jointfm
    ):
        config = _config(tmp_path, prices_path)
        first = run_experiment(config, command=["first"])
        calls = len(fake_jointfm.forecast_calls)

        again = run_experiment(config, command=["again"])
        assert len(fake_jointfm.forecast_calls) == calls
        pd.testing.assert_frame_equal(again, first)

        extended = run_experiment(
            _config(tmp_path, prices_path, trials=3), command=["extended"]
        )
        assert len(extended) == 3 * N_ORIGINS * len(CPU_ARMS)
        assert not extended.duplicated(["trial", "origin", "arm"]).any()
        pd.testing.assert_frame_equal(extended.iloc[: len(first)], first)
        _, metadata = read_results(config.output)
        # A session with nothing left to run writes nothing.
        assert metadata.commands == [["first"], ["extended"]]

    def test_only_first_call_per_arm_and_process_is_warmup(
        self, tmp_path, prices_path, fake_jointfm
    ):
        frame = run_experiment(_config(tmp_path, prices_path, trials=1))
        warmup = frame[frame["warmup"]]
        assert sorted(warmup["arm"]) == sorted(CPU_ARMS)
        assert (warmup["origin"] == frame["origin"].min()).all()

        extended = run_experiment(_config(tmp_path, prices_path, trials=2))
        new = extended[extended["trial"] == 1]
        assert sorted(new.loc[new["warmup"], "arm"]) == sorted(CPU_ARMS)
        assert extended["warmup"].sum() == 2 * len(CPU_ARMS)

    def test_results_file_records_the_run(self, tmp_path, prices_path, fake_jointfm):
        config = _config(tmp_path, prices_path)
        run_experiment(config)
        _, metadata = read_results(config.output)
        identity = metadata.identity
        assert identity["experiment"]["num_scen"] == 16
        assert set(identity["arms"]) == set(CPU_ARMS)
        assert identity["arms"]["kde-cpu"]["kde_settings"]["device"] == "CPU"
        assert identity["services"]["jointfm-cpu"]["device"] == "cpu"
        assert identity["services"]["jointfm-cpu"]["client_version"] == "0.0.fake"
        assert identity["host"]["gpu"] is None
        assert "pyarrow" in identity["packages"]

    def test_other_conditions_are_refused(self, tmp_path, prices_path, fake_jointfm):
        run_experiment(_config(tmp_path, prices_path))
        with pytest.raises(ValueError, match="experiment"):
            run_experiment(_config(tmp_path, prices_path, num_scen=32))

    def test_jointfm_arm_reads_dotenv_like_the_library(
        self, tmp_path, prices_path, fake_jointfm
    ):
        run_experiment(_config(tmp_path, prices_path, trials=1))
        assert fake_jointfm.from_env_kwargs == {
            "dotenv_path": ".env",
            "config_path": "config.yaml",
        }

    def test_second_device_adds_its_jointfm_arm(
        self, tmp_path, prices_path, fake_jointfm
    ):
        config = _config(tmp_path, prices_path)
        first = run_experiment(config)
        fake_jointfm.device = "cuda:0"
        calls = len(fake_jointfm.forecast_calls)

        both = run_experiment(config)
        added = both.iloc[len(first) :]
        assert set(added["arm"]) == {"jointfm-gpu"}
        assert len(added) == 2 * N_ORIGINS
        assert len(fake_jointfm.forecast_calls) == calls + len(added)
        _, metadata = read_results(config.output)
        assert set(metadata.identity["arms"]) == {*CPU_ARMS, "jointfm-gpu"}
        assert {
            arm: service["device"]
            for arm, service in metadata.identity["services"].items()
        } == {"jointfm-cpu": "cpu", "jointfm-gpu": "cuda:0"}

    def test_second_device_must_serve_the_same_checkpoint(
        self, tmp_path, prices_path, fake_jointfm
    ):
        config = _config(tmp_path, prices_path)
        run_experiment(config)
        fake_jointfm.device = "cuda:0"
        fake_jointfm.checkpoint_version = "other-checkpoint"
        with pytest.raises(ValueError, match="different checkpoints"):
            run_experiment(config)

    def test_unknown_service_device_is_refused(
        self, tmp_path, prices_path, fake_jointfm
    ):
        fake_jointfm.device = "mps"
        config = _config(tmp_path, prices_path)
        with pytest.raises(ValueError, match="'mps'"):
            run_experiment(config)
        assert not config.output.exists()

    def test_measurement_must_return_its_columns(
        self, tmp_path, prices_path, fake_jointfm
    ):
        class Wrong(RecordingMeasurement):
            def __call__(self, inputs, horizon, samples, expected_returns):
                return {"other": 1.0}

        with pytest.raises(ValueError, match="declared"):
            run_experiment(_config(tmp_path, prices_path), [Wrong()])

    def test_scores_fill_every_row(self, tmp_path, prices_path, fake_jointfm):
        config = _config(tmp_path, prices_path, trials=1)
        portfolios = PortfolioScores(
            config.portfolio_sizes, config.rebalancing.cvar, config.assets
        )
        frame = run_experiment(config, [ScenarioScores(), portfolios])
        scores = frame[[*ScenarioScores.columns, *portfolios.columns]]
        assert np.isfinite(scores.to_numpy()).all()
        no_fit = frame[frame["arm"] == "no_fit-cpu"]
        assert (no_fit["corr_error_trailing"] == 0.0).all()
        _, metadata = read_results(config.output)
        recorded, recorded_portfolios = metadata.identity["measurements"]
        assert recorded["columns"] == list(ScenarioScores.columns)
        assert recorded["parameters"] == ScenarioScores.parameters
        assert recorded_portfolios["columns"] == list(portfolios.columns)
        assert recorded_portfolios["parameters"]["cells"] == [
            ["equal", 1],
            ["long_only", 1],
            ["equal", 2],
            ["long_only", 2],
            ["equal", 4],
            ["long_only", 4],
        ]

    def test_every_horizon_gets_a_row_from_one_call(
        self, tmp_path, prices_path, fake_jointfm
    ):
        """Each arm's call at an origin writes one timed row per horizon."""
        fake_jointfm.envelope.n_input = 1000
        config = _config(tmp_path, prices_path, trials=1, horizons=(1, 3, FORWARD))
        frame = run_experiment(config, [ScenarioScores()])

        assert len(frame) == N_ORIGINS * len(CPU_ARMS) * 3
        assert not frame.duplicated(["trial", "origin", "arm", "horizon"]).any()
        calls = frame.groupby(["trial", "origin", "arm"])
        assert all(
            horizons == [1, 3, FORWARD] for horizons in calls["horizon"].apply(list)
        )
        # One timed call generates every horizon of an arm at an origin.
        assert (calls["seconds"].nunique() == 1).all()
        assert (calls["warmup"].nunique() == 1).all()
        # JointFM answers every horizon in one request.
        assert len(fake_jointfm.forecast_calls) == N_ORIGINS
        for call in fake_jointfm.forecast_calls:
            last = len(call["history"]) - 1
            assert call["query_times"] == [last + 1, last + 3, last + FORWARD]
        no_fit = frame[frame["arm"] == "no_fit-cpu"].set_index("horizon")["num_scen"]
        assert (no_fit.loc[1] == LOOK_BACK).all()
        assert (no_fit.loc[[3, FORWARD]] == 16).all()
        _, metadata = read_results(config.output)
        assert metadata.identity["experiment"]["horizons"] == [1, 3, FORWARD]

    def test_next_day_rows_do_not_depend_on_other_horizons(
        self, tmp_path, prices_path, fake_jointfm
    ):
        """Adding horizons leaves the simulators' next-day readings unchanged."""
        fake_jointfm.envelope.n_input = 1000
        config = _config(tmp_path, prices_path, trials=1)
        portfolios = PortfolioScores(
            config.portfolio_sizes, config.rebalancing.cvar, config.assets
        )
        measurements = [ScenarioScores(), portfolios]
        alone = run_experiment(config, measurements)
        several = run_experiment(
            _config(
                tmp_path,
                prices_path,
                trials=1,
                horizons=(1, 3, FORWARD),
                output=tmp_path / "several.parquet",
            ),
            measurements,
        )
        scores = [*ScenarioScores.columns, *portfolios.columns]
        next_day = several[several["horizon"] == 1].reset_index(drop=True)
        assert np.isfinite(several[scores].to_numpy()).all()
        # The fake service draws all horizons at once, so only the simulators
        # the comparison runs itself reproduce their next day exactly.
        exact = alone["fit_type"] != "jointfm"
        pd.testing.assert_frame_equal(
            next_day.loc[exact, ["arm", "origin", "num_scen", *scores]],
            alone.loc[exact, ["arm", "origin", "num_scen", *scores]],
        )


class TestJointFMService:
    def test_jointfm_arm_uses_its_own_scenario_mean(
        self, tmp_path, prices_path, fake_jointfm
    ):
        config = _config(tmp_path, prices_path, trials=1)
        run_experiment(config)
        _, metadata = read_results(config.output)
        settings = metadata.identity["arms"]["jointfm-cpu"]["jointfm_settings"]
        assert settings["expected_return"] == "scenario"

    def test_service_switch_during_a_session_is_refused(
        self, tmp_path, prices_path, fake_jointfm
    ):
        class SwitchService(RecordingMeasurement):
            """Points .env at a GPU service once the first origin's arms ran."""

            def __call__(self, inputs, horizon, samples, expected_returns):
                values = super().__call__(inputs, horizon, samples, expected_returns)
                if len(self.calls) == len(CPU_ARMS):
                    fake_jointfm.device = "cuda:0"
                return values

        config = _config(tmp_path, prices_path)
        with pytest.raises(ValueError, match="changed during the jointfm-cpu"):
            run_experiment(config, [SwitchService()])
        frame, _ = read_results(config.output)
        assert sorted(frame["arm"]) == sorted(CPU_ARMS)

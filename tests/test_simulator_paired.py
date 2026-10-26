# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the origin history and the paired run comparison in
experiments/, with CLARABEL and the fake JointFM client of conftest.py."""

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from experiments.simulator_comparison.arms import ARMS_BY_NAME
from experiments.simulator_comparison.config import ExperimentConfig
from experiments.simulator_comparison.paired import (
    REBALANCING,
    SCENARIO,
    main,
    paired_comparison,
    write_paired_comparison,
)
from experiments.simulator_comparison.rebalancing import run_rebalancing_experiment
from experiments.simulator_comparison.runner import pending_work, run_experiment
from experiments.simulator_comparison.scores import (
    PortfolioScores,
    ScenarioScores,
    cell_column,
)
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
HORIZONS = (1, 4)
ROWS = 300
ASSETS = 4
SCENARIO_TRIALS = 4
REBALANCING_TRIALS = 2
CONFIG_ARMS = ("no_fit-cpu", "jointfm")
RUN_ARMS = ("no_fit-cpu", "jointfm-cpu")
JOINTFM = "jointfm-cpu"
RATES_FILE = "rates.csv"
BASE_RISK_AVERSION = 1.0
HIGH_RISK_AVERSION = 100.0
# Planted per-trial change of the JointFM arm's reading in a copied results file.
PLANTED_CHANGE = 0.25
PLANTED_READING = {SCENARIO: "crps", REBALANCING: "sharpe_ratio"}
# A full-universe equal-weight portfolio cell of the test configs.
EQUAL_BREACH = cell_column("equal", ASSETS, "breach_05")


def _write_prices(directory):
    """Prices with positive drift, which keeps the risk-aversion calibration
    defined, and daily rates over their span."""
    rng = np.random.default_rng(3)
    dates = pd.date_range("2019-01-01", periods=ROWS, freq="B")
    log_prices = np.cumsum(rng.normal(0.001, 0.01, size=(ROWS, 6)), axis=0)
    prices = pd.DataFrame(
        100.0 * np.exp(log_prices), index=dates, columns=[f"A{i}" for i in range(6)]
    )
    path = directory / "prices.csv"
    prices.to_csv(path)
    rate_dates = pd.date_range("2018-12-03", "2020-06-30", freq="B")
    rates = pd.Series(np.linspace(0.01, 0.05, len(rate_dates)), index=rate_dates)
    rates.rename("rate").rename_axis("Date").to_csv(directory / RATES_FILE)
    return path


def _config(
    directory,
    prices_path,
    name,
    risk_aversion=BASE_RISK_AVERSION,
    **overrides,
):
    output = directory / f"results_{name}.parquet"
    values = dict(
        prices=prices_path,
        output=output,
        trials=SCENARIO_TRIALS,
        origins=3,
        assets=ASSETS,
        look_back=SHORT_LOOK_BACK,
        origin_history=LONG_LOOK_BACK,
        forward=FORWARD,
        num_scen=16,
        seed=5,
        arms=CONFIG_ARMS,
        portfolio_sizes=(1, ASSETS),
        horizons=HORIZONS,
        rebalancing=dict(
            output=directory / f"rebalancing_{name}.parquet",
            scenario_results=output,
            arms=CONFIG_ARMS,
            trials=REBALANCING_TRIALS,
            trading_days=10,
            look_forward_window=5,
            look_back_days=90,
            trigger="every_step",
            pct_change_threshold=-0.005,
            transaction_cost=0.001,
            risk_free_rates=directory / RATES_FILE,
            solver="CLARABEL",
            cvar=dict(
                w_min=0.0,
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


@dataclass(frozen=True)
class Runs:
    """A base config and its twins, each differing in one setting."""

    baseline: ExperimentConfig
    long_history: ExperimentConfig
    next_day: ExperimentConfig
    risk_averse: ExperimentConfig
    # Lengths of the histories JointFM received, per look_back.
    history_lengths: dict[int, set[int]]


@pytest.fixture(scope="module")
def runs(tmp_path_factory, module_fake_jointfm):
    """The base config's scenario and rebalancing results and those of its
    long-history, next-day, and risk-averse twins, run once on one fake service."""
    module_fake_jointfm.envelope.n_input = 1000
    directory = tmp_path_factory.mktemp("paired")
    prices_path = _write_prices(directory)
    baseline = _config(directory, prices_path, "baseline")
    long_history = _config(
        directory, prices_path, "long_history", look_back=LONG_LOOK_BACK
    )
    next_day = _config(directory, prices_path, "next_day", horizons=(1,))
    risk_averse = _config(
        directory, prices_path, "risk_averse", risk_aversion=HIGH_RISK_AVERSION
    )
    history_lengths = {}
    for config in (baseline, long_history):
        module_fake_jointfm.forecast_calls.clear()
        _run(config)
        history_lengths[config.look_back] = {
            len(call["history"]) for call in module_fake_jointfm.forecast_calls
        }
    _run(next_day)
    run_rebalancing_experiment(baseline)
    run_rebalancing_experiment(risk_averse)
    return Runs(baseline, long_history, next_day, risk_averse, history_lengths)


def _results_path(config, results):
    return config.output if results == SCENARIO else config.rebalancing.output


def _copy_results(source, target, edit_identity, edit_frame=None):
    """Write ``source``'s rows to ``target`` with an edited identity and frame.

    A scenario file keeps the rows of the horizons its edited identity holds.
    """
    frame, metadata = read_results(source)
    identity = copy.deepcopy(metadata.identity)
    edit_identity(identity)
    if "horizon" in frame:
        frame = frame[frame["horizon"].isin(identity["experiment"]["horizons"])]
    frame = frame.reset_index(drop=True)
    if edit_frame is not None:
        edit_frame(frame)
    write_results(target, frame, RunMetadata(identity, metadata.commands))
    return target


def _unchanged(identity):
    """Leave the identity as it is, so both files hold one value of every setting."""


def _other_look_back(identity):
    identity["experiment"]["look_back"] = LONG_LOOK_BACK


def _next_day_only(identity):
    identity["experiment"]["horizons"] = [1]


def _other_risk_aversion(identity):
    identity["rebalancing"]["cvar"]["risk_aversion"] = 2.0


def _other_checkpoint(identity):
    for service in identity["services"].values():
        service["checkpoint_version"] = "other-checkpoint"


def _other_device(identity):
    """The JointFM service on the other device, whose arm the baseline lacks."""
    identity["services"] = {"jointfm-gpu": identity["services"][JOINTFM]}


def _other_num_scen(identity):
    identity["experiment"]["num_scen"] = 8


def _other_origin_history(identity):
    identity["experiment"]["origin_history"] = 1


def _other_transaction_cost(identity):
    identity["rebalancing"]["transaction_cost"] = 0.5


def _combined(*edits):
    """One identity edit that applies ``edits`` in turn."""

    def edit_all(identity):
        for edit in edits:
            edit(identity)

    return edit_all


def _plant(column, trials=None):
    """Add ``PLANTED_CHANGE`` to the JointFM arm's ``column``, in ``trials`` or all."""

    def plant(frame):
        rows = frame["arm"] == JOINTFM
        if trials is not None:
            rows &= frame["trial"].isin(trials)
        frame.loc[rows, column] += PLANTED_CHANGE

    return plant


class TestOriginHistory:
    @pytest.fixture()
    def prices_path(self, tmp_path):
        return _write_prices(tmp_path)

    def test_origins_do_not_depend_on_look_back(self, tmp_path, prices_path):
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        arms = [ARMS_BY_NAME["no_fit-cpu"]]
        drawn = [
            [
                (trial, origin)
                for trial, origin, _ in pending_work(
                    prices,
                    _config(tmp_path, prices_path, "any", look_back=look_back),
                    arms,
                    set(),
                )
            ]
            for look_back in (SHORT_LOOK_BACK, LONG_LOOK_BACK)
        ]
        assert drawn[0] == drawn[1]
        assert min(o for _, o in drawn[0]) >= LONG_LOOK_BACK

    def test_every_window_holds_look_back_returns(self, prices_path):
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        trial = draw_trial(
            prices,
            0,
            seed=5,
            n_assets=ASSETS,
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
            _config(tmp_path, prices_path, "any", origin_history=63)


class TestTwins:
    """The shipped twin pattern: each twin's files pair with its base's."""

    def test_look_back_twins_pair_by_trial(self, runs):
        keys = [
            read_results(config.output)[0][["trial", "origin", "seed"]]
            .drop_duplicates()
            .reset_index(drop=True)
            for config in (runs.baseline, runs.long_history)
        ]
        pd.testing.assert_frame_equal(keys[0], keys[1])
        # A window of n returns is sent as its n + 1 log prices.
        assert runs.history_lengths == {
            SHORT_LOOK_BACK: {SHORT_LOOK_BACK + 1},
            LONG_LOOK_BACK: {LONG_LOOK_BACK + 1},
        }
        comparison = paired_comparison(
            runs.baseline.output, runs.long_history.output, "look_back", SCENARIO
        )
        table = comparison.table
        assert (comparison.baseline_value, comparison.value) == ("64", "80")
        assert set(table["arm"]) == set(RUN_ARMS)
        assert set(table["horizon"]) == set(HORIZONS)
        assert (table["trials"] == SCENARIO_TRIALS).all()
        # no_fit's next-day scenarios are its window, zero bias at any length.
        no_fit_bias = table[
            (table["arm"] == "no_fit-cpu")
            & (table["metric"] == "corr_bias_trailing")
            & (table["horizon"] == 1)
        ]
        assert no_fit_bias["difference"].item() == 0.0

    def test_horizons_twins_pair_at_the_next_day_where_the_baselines_are_unchanged(
        self, runs
    ):
        table = paired_comparison(
            runs.baseline.output, runs.next_day.output, "horizons", SCENARIO
        ).table
        assert set(table["horizon"]) == {1}
        assert set(table["value"]) == {"[1]"}
        assert set(table["baseline_value"]) == {"[1, 4]"}
        no_fit = table[table["arm"] == "no_fit-cpu"]
        assert len(no_fit) and (no_fit["difference"] == 0.0).all()
        assert JOINTFM in set(table["arm"])

    def test_risk_aversion_twins_pair_although_their_scenario_results_differ(
        self, runs
    ):
        assert (
            runs.baseline.rebalancing.scenario_results
            != runs.risk_averse.rebalancing.scenario_results
        )
        table = paired_comparison(
            runs.baseline.rebalancing.output,
            runs.risk_averse.rebalancing.output,
            "risk_aversion",
            REBALANCING,
        ).table.set_index(["arm", "metric"])
        assert set(table["value"]) == {str(HIGH_RISK_AVERSION)}
        assert (table["trials"] == REBALANCING_TRIALS).all()
        assert table.loc[(JOINTFM, "chosen_breach_rate"), "statistic"] == "mean"
        assert table.loc[(JOINTFM, "sharpe_ratio"), "statistic"] == "median"


PLANTED_CASES = [
    pytest.param("look_back", SCENARIO, _other_look_back, id="look_back"),
    pytest.param("horizons", SCENARIO, _next_day_only, id="horizons"),
    pytest.param(
        "risk_aversion", REBALANCING, _other_risk_aversion, id="risk_aversion"
    ),
    pytest.param("checkpoint", SCENARIO, _other_checkpoint, id="checkpoint scenario"),
    pytest.param(
        "checkpoint", REBALANCING, _other_checkpoint, id="checkpoint rebalancing"
    ),
]


class TestPairedComparison:
    @pytest.mark.parametrize("setting, results, edit", PLANTED_CASES)
    def test_a_planted_change_is_recovered_and_drawn(
        self, runs, tmp_path, setting, results, edit
    ):
        reading = PLANTED_READING[results]
        baseline = _results_path(runs.baseline, results)
        other = _copy_results(
            baseline, tmp_path / "other.parquet", edit, _plant(reading)
        )
        comparison = paired_comparison(baseline, other, setting, results)
        table = comparison.table
        planted = (table["arm"] == JOINTFM) & (table["metric"] == reading)
        assert planted.any()
        for column in ("difference", "ci_low", "ci_high"):
            np.testing.assert_allclose(table.loc[planted, column], PLANTED_CHANGE)
        assert (table.loc[planted, "statistic"] == "median").all()
        # Nothing else moved: every other arm and reading pairs identical rows.
        assert (table.loc[~planted, "difference"] == 0.0).all()

        files = write_paired_comparison(comparison, tmp_path / "report")
        names = {path.name for path in files}
        figures = [f"paired_{setting}"]
        if results == SCENARIO:
            figures.append(f"paired_{setting}_portfolios")
        assert f"paired_{setting}.csv" in names
        for figure in figures:
            assert {f"{figure}.png", f"{figure}.svg"} <= names
        assert all(path.exists() for path in files)

    def test_a_share_of_events_is_compared_by_its_mean(self, runs, tmp_path):
        # A change in one of four trials: the mean keeps it, the median reads 0.
        other = _copy_results(
            runs.baseline.output,
            tmp_path / "other.parquet",
            _other_look_back,
            _plant(["ew_var_exceeded", EQUAL_BREACH, "crps"], trials=[0]),
        )
        table = (
            paired_comparison(runs.baseline.output, other, "look_back", SCENARIO)
            .table.query("horizon == 1")
            .set_index(["arm", "metric"])
        )
        for share in ("ew_var_exceeded", EQUAL_BREACH):
            row = table.loc[(JOINTFM, share)]
            assert row["statistic"] == "mean"
            assert row["difference"] == pytest.approx(PLANTED_CHANGE / SCENARIO_TRIALS)
        crps = table.loc[(JOINTFM, "crps")]
        assert crps["statistic"] == "median"
        assert crps["difference"] == 0.0

    def test_the_same_checkpoint_label_is_compared(self, runs, tmp_path):
        other = _copy_results(
            runs.baseline.output, tmp_path / "other.parquet", _unchanged
        )
        comparison = paired_comparison(
            runs.baseline.output, other, "checkpoint", SCENARIO
        )
        assert (
            comparison.value
            == comparison.baseline_value
            == "fake-model fake-checkpoint"
        )
        assert (comparison.table["difference"] == 0.0).all()

    def test_another_host_is_accepted(self, runs, tmp_path):
        def other_host(identity):
            identity["host"]["hostname"] = "another-host"

        other = _copy_results(
            runs.baseline.output,
            tmp_path / "host.parquet",
            _combined(_other_look_back, other_host),
        )
        assert len(
            paired_comparison(runs.baseline.output, other, "look_back", SCENARIO).table
        )

    @pytest.mark.parametrize(
        "setting, results, edit, message",
        [
            ("look_back", SCENARIO, _unchanged, "Both files hold look_back"),
            ("horizons", SCENARIO, _unchanged, "Both files hold horizons"),
            ("risk_aversion", REBALANCING, _unchanged, "Both files hold risk_aversion"),
            (
                "look_back",
                SCENARIO,
                _combined(_other_look_back, _other_num_scen),
                "'num_scen': 8",
            ),
            (
                "horizons",
                SCENARIO,
                _combined(_next_day_only, _other_origin_history),
                "'origin_history': 1",
            ),
            (
                "risk_aversion",
                REBALANCING,
                _combined(_other_risk_aversion, _other_transaction_cost),
                "'transaction_cost': 0.5",
            ),
            (
                "checkpoint",
                SCENARIO,
                _combined(_other_checkpoint, _other_num_scen),
                "'num_scen': 8",
            ),
            (
                "checkpoint",
                REBALANCING,
                _combined(_other_checkpoint, _other_transaction_cost),
                "'transaction_cost': 0.5",
            ),
            (
                "look_back",
                SCENARIO,
                _combined(_other_look_back, _other_checkpoint),
                f"services.{JOINTFM}",
            ),
            ("checkpoint", SCENARIO, _other_device, "share no JointFM arm"),
        ],
        ids=[
            "same look_back",
            "same horizons",
            "same risk_aversion",
            "look_back and num_scen",
            "horizons and origin_history",
            "risk_aversion and transaction_cost",
            "checkpoint and num_scen",
            "checkpoint and transaction_cost",
            "look_back and checkpoint",
            "checkpoint on another device",
        ],
    )
    def test_other_conditions_are_refused(
        self, runs, tmp_path, setting, results, edit, message
    ):
        baseline = _results_path(runs.baseline, results)
        other = _copy_results(baseline, tmp_path / "other.parquet", edit)
        with pytest.raises(ValueError, match=message):
            paired_comparison(baseline, other, setting, results)

    @pytest.mark.parametrize(
        "setting, results, message",
        [
            ("risk_aversion", SCENARIO, "changes the rebalancing results"),
            ("look_back", REBALANCING, "changes the scenario results"),
            ("seed", SCENARIO, "Unknown setting 'seed'"),
        ],
        ids=["risk_aversion of scenarios", "look_back of rebalancing", "unknown"],
    )
    def test_a_setting_reads_only_its_results(self, runs, setting, results, message):
        path = _results_path(runs.baseline, results)
        with pytest.raises(ValueError, match=message):
            paired_comparison(path, path, setting, results)

    def test_files_of_other_results_are_refused(self, runs):
        with pytest.raises(ValueError, match="holds rebalancing results"):
            paired_comparison(
                runs.baseline.output,
                runs.baseline.rebalancing.output,
                "checkpoint",
                SCENARIO,
            )


def _write_config(config, path):
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    return path


@pytest.mark.parametrize(
    "twin, setting, results",
    [
        ("long_history", "look_back", SCENARIO),
        ("risk_averse", "risk_aversion", REBALANCING),
    ],
)
def test_the_command_writes_into_the_other_config_report(
    runs, tmp_path, capsys, twin, setting, results
):
    other = getattr(runs, twin)
    argv = [
        "--baseline",
        str(_write_config(runs.baseline, tmp_path / "baseline.yaml")),
        "--config",
        str(_write_config(other, tmp_path / "other.yaml")),
        "--setting",
        setting,
        "--results",
        results,
    ]
    assert main(argv) == 0
    report = other.report_dir if results == SCENARIO else other.rebalancing.report_dir
    assert (report / f"paired_{setting}.csv").exists()
    assert (report / f"paired_{setting}.png").exists()
    assert "Paired comparison finished" in capsys.readouterr().out

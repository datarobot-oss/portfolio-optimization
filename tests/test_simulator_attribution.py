# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rebalancing attribution of the simulator comparison in experiments/: the
historical-mean JointFM arm, the independence control, the expected-return
reading, and the split of a difference into an expected-return and a scenario
part, with the fake JointFM client of conftest.py."""

import numpy as np
import pandas as pd
import pytest

from experiments.simulator_comparison import rebalancing_report
from experiments.simulator_comparison.analysis import (
    INDEPENDENCE_EFFECT,
    independence_design,
    independence_effects,
    mean_attribution,
    mean_attribution_design,
)
from experiments.simulator_comparison.arms import (
    ARMS_BY_NAME,
    DERIVED_DONORS,
    HISTORICAL_MEAN_ARM,
    INDEPENDENT_ARM,
    JOINTFM_ARM,
    SWAP_DONORS,
    WINDOW_ARM,
    HistoricalMeanArm,
    IndependentArm,
    scenario_settings,
)
from experiments.simulator_comparison.config import ExperimentConfig
from experiments.simulator_comparison.independence import (
    IndependentScenarioGenerator,
    independence_rng,
    permute_assets,
)
from experiments.simulator_comparison.plotting import (
    ATTRIBUTION_STYLES,
    SIMULATOR_COLORS,
)
from experiments.simulator_comparison.rebalancing import (
    METRICS,
    backtest_scenarios,
    run_rebalancing_experiment,
)
from experiments.simulator_comparison.rebalancing_report import (
    METRIC_LABELS,
    write_rebalancing_report,
)
from experiments.simulator_comparison.report import write_report
from experiments.simulator_comparison.runner import run_experiment
from experiments.simulator_comparison.scores import (
    PortfolioScores,
    ScenarioScores,
    expected_return_correlation,
)
from experiments.simulator_comparison.table import read_results
from experiments.simulator_comparison.trials import RETURNS_SETTINGS
from portfolio_optimization.cvar_utils import generate_cvar_data
from portfolio_optimization.utils import calculate_returns

ASSETS = 4
NUM_SCEN = 100
HORIZONS = (1, 3)
# The historical-mean arm runs only in the rebalancing backtest.
COMPARISON_ARMS = (WINDOW_ARM, JOINTFM_ARM, INDEPENDENT_ARM)
REBALANCING_ARMS = (*COMPARISON_ARMS, HISTORICAL_MEAN_ARM)
JOINTFM_CPU = "jointfm-cpu"


def _correlated(rng, rows, rho, assets=ASSETS):
    """Gaussian returns whose assets share one factor with correlation ``rho``."""
    factor = rng.normal(size=(rows, 1))
    noise = rng.normal(size=(rows, assets))
    return np.sqrt(rho) * factor + np.sqrt(1 - rho) * noise


def _mean_off_diagonal_correlation(samples):
    correlations = np.corrcoef(samples.T)
    return correlations[np.triu_indices(samples.shape[1], 1)].mean()


@pytest.fixture()
def prices_path(tmp_path):
    """Correlated random-walk prices with positive drift, and their cash rates."""
    rows = 300
    rng = np.random.default_rng(12)
    dates = pd.date_range("2019-01-01", periods=rows, freq="B")
    returns = 0.001 + 0.01 * _correlated(rng, rows, 0.6, ASSETS + 2)
    frame = pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=[f"A{i}" for i in range(ASSETS + 2)],
    )
    path = tmp_path / "prices.csv"
    frame.to_csv(path)
    rate_dates = pd.date_range("2018-12-03", "2020-03-31", freq="B")
    pd.Series(0.02, index=rate_dates, name="rate").rename_axis("Date").to_csv(
        path.with_name("rates.csv")
    )
    return path


def _config(
    tmp_path,
    prices_path,
    arms=COMPARISON_ARMS,
    rebalancing_arms=REBALANCING_ARMS,
    **overrides,
):
    """A small comparison of ``arms`` whose rebalancing runs ``rebalancing_arms``."""
    values = dict(
        prices=prices_path,
        output=tmp_path / "results.parquet",
        trials=2,
        origins=3,
        assets=ASSETS,
        look_back=64,
        origin_history=64,
        forward=8,
        num_scen=NUM_SCEN,
        seed=5,
        arms=arms,
        portfolio_sizes=(1, ASSETS),
        horizons=HORIZONS,
        rebalancing=dict(
            output=tmp_path / "rebalancing.parquet",
            scenario_results=tmp_path / "results.parquet",
            arms=rebalancing_arms,
            trials=2,
            trading_days=10,
            look_forward_window=5,
            look_back_days=90,
            trigger="every_step",
            pct_change_threshold=-0.005,
            transaction_cost=0.001,
            risk_free_rates=prices_path.with_name("rates.csv"),
            solver="CLARABEL",
            cvar=dict(
                w_min=0.0,
                w_max=0.6,
                c_min=0.0,
                c_max=0.4,
                L_tar=1.6,
                T_tar=1.0,
                confidence=0.95,
                risk_aversion=1.0,
            ),
        ),
    )
    values.update(overrides)
    return ExperimentConfig(**values)


def _returns_dict(prices_path):
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    return calculate_returns(
        prices.iloc[:65, :ASSETS], returns_compute_settings=RETURNS_SETTINGS
    )


class TestIndependenceControl:
    def test_keeps_every_assets_returns_and_destroys_their_alignment(self):
        window = _correlated(np.random.default_rng(1), 2000, 0.8)
        permuted = permute_assets(window, independence_rng(7, 1))
        np.testing.assert_array_equal(
            np.sort(permuted, axis=0), np.sort(window, axis=0)
        )
        np.testing.assert_allclose(permuted.mean(axis=0), window.mean(axis=0))
        assert _mean_off_diagonal_correlation(window) > 0.75
        assert abs(_mean_off_diagonal_correlation(permuted)) < 0.05

    def test_assets_almost_never_share_a_window_row(self):
        """Each scenario takes every asset's value from an unrelated window row."""
        rows = np.tile(np.arange(2000.0)[:, None], (1, ASSETS))
        permuted = permute_assets(rows, independence_rng(7, 1))
        for asset in range(1, ASSETS):
            assert np.mean(permuted[:, asset] == permuted[:, 0]) < 0.01

    def test_permutation_is_reproducible_and_seeded_per_horizon(self):
        window = _correlated(np.random.default_rng(2), 50, 0.5)
        first = permute_assets(window, independence_rng(3, 1))
        np.testing.assert_array_equal(
            first, permute_assets(window, independence_rng(3, 1))
        )
        assert not np.array_equal(first, permute_assets(window, independence_rng(3, 2)))

    def test_refuses_scenarios_that_are_not_a_matrix(self):
        with pytest.raises(ValueError, match="scenarios, assets"):
            permute_assets(np.zeros(5), independence_rng(0, 1))

    def test_generator_keeps_the_windows_returns_and_mean(self, prices_path):
        returns_dict = _returns_dict(prices_path)
        arm = IndependentArm(INDEPENDENT_ARM, ARMS_BY_NAME[WINDOW_ARM])
        settings, generator = backtest_scenarios(arm, num_scen=NUM_SCEN, seed=3)
        assert settings == scenario_settings(
            ARMS_BY_NAME[WINDOW_ARM], num_scen=NUM_SCEN, seed=3
        )
        control = generator(dict(returns_dict), settings, cash_return=0.001)[
            "cvar_data"
        ]
        window = generate_cvar_data(dict(returns_dict), settings)["cvar_data"]
        np.testing.assert_array_equal(control.mean, window.mean)
        np.testing.assert_array_equal(
            control.R.T, permute_assets(window.R.T, independence_rng(3, 1))
        )
        np.testing.assert_array_equal(
            np.sort(control.R, axis=1), np.sort(window.R, axis=1)
        )
        np.testing.assert_array_equal(control.p, window.p)
        assert control.cash_return == 0.001

    def test_generator_needs_a_seed(self, prices_path):
        settings = scenario_settings(ARMS_BY_NAME[WINDOW_ARM], num_scen=1, seed=None)
        with pytest.raises(ValueError, match="seeded"):
            IndependentScenarioGenerator()(_returns_dict(prices_path), settings)


class TestHistoricalMeanArm:
    def test_draws_jointfms_scenarios_on_the_windows_mean(
        self, prices_path, fake_jointfm
    ):
        fake_jointfm.envelope.n_input = 1000
        returns_dict = _returns_dict(prices_path)
        jointfm = ARMS_BY_NAME[JOINTFM_CPU]
        arm = HistoricalMeanArm(HISTORICAL_MEAN_ARM, jointfm)
        settings, generator = backtest_scenarios(arm, num_scen=NUM_SCEN, seed=3)
        assert settings.jointfm_settings.expected_return == "historical"
        donor = scenario_settings(jointfm, num_scen=NUM_SCEN, seed=3)
        assert settings.model_dump(exclude={"jointfm_settings"}) == donor.model_dump(
            exclude={"jointfm_settings"}
        )
        historical = generator(dict(returns_dict), settings)["cvar_data"]
        scenario = generate_cvar_data(dict(returns_dict), donor)["cvar_data"]
        np.testing.assert_array_equal(historical.R, scenario.R)
        np.testing.assert_array_equal(historical.mean, returns_dict["mean"])
        np.testing.assert_allclose(scenario.mean, scenario.R.mean(axis=1))

    def test_refuses_a_donor_other_than_jointfm(self):
        with pytest.raises(ValueError, match="JointFM arm"):
            HistoricalMeanArm(HISTORICAL_MEAN_ARM, ARMS_BY_NAME[WINDOW_ARM])

    @pytest.mark.parametrize(
        ("arms", "missing"),
        [
            ((WINDOW_ARM, HISTORICAL_MEAN_ARM), JOINTFM_ARM),
            ((JOINTFM_ARM, INDEPENDENT_ARM), WINDOW_ARM),
        ],
    )
    def test_derived_arms_need_their_donors(self, tmp_path, prices_path, arms, missing):
        with pytest.raises(ValueError, match=f"next to their donors.*{missing}"):
            _config(tmp_path, prices_path, arms=(WINDOW_ARM,), rebalancing_arms=arms)

    def test_runs_only_in_the_rebalancing_backtest(self, tmp_path, prices_path):
        """Its scenario-comparison rows would repeat JointFM's and the window's."""
        with pytest.raises(ValueError, match="only in rebalancing.arms"):
            _config(tmp_path, prices_path, arms=REBALANCING_ARMS)


def test_every_new_arm_has_a_figure_color():
    assert {HISTORICAL_MEAN_ARM, INDEPENDENT_ARM} <= set(SIMULATOR_COLORS)


class TestExpectedReturnCorrelation:
    def test_reads_the_cross_asset_correlation(self):
        realized = np.array([0.01, -0.02, 0.03, 0.0])
        assert expected_return_correlation(2 * realized + 1, realized) == pytest.approx(
            1.0
        )
        assert expected_return_correlation(-realized, realized) == pytest.approx(-1.0)

    @pytest.mark.parametrize(
        ("expected", "realized", "message"),
        [
            (np.zeros(4), np.arange(4.0), "undefined"),
            (np.arange(4.0), np.ones(4), "undefined"),
            (np.arange(3.0), np.arange(4.0), "shape"),
        ],
    )
    def test_refuses_undefined_correlations(self, expected, realized, message):
        with pytest.raises(ValueError, match=message):
            expected_return_correlation(expected, realized)


class RecordingForecasts:
    """Records every call's scenarios and expected returns, in call order."""

    columns = ("rows",)
    parameters = {}

    def __init__(self):
        self.calls = []

    def __call__(self, inputs, horizon, samples, expected_returns):
        self.calls.append((inputs, horizon, samples, expected_returns))
        return {"rows": float(samples.shape[0])}


class TestComparison:
    def test_arms_reuse_their_donors_and_read_their_expected_returns(
        self, tmp_path, prices_path, fake_jointfm
    ):
        fake_jointfm.envelope.n_input = 1000
        recording = RecordingForecasts()
        config = _config(tmp_path, prices_path)
        frame = run_experiment(config, [recording])

        assert len(fake_jointfm.forecast_calls) == config.trials * config.origins
        calls = {
            (row.arm, row.seed, row.horizon): call
            for row, call in zip(frame.itertuples(), recording.calls)
        }
        for (arm, seed, horizon), (inputs, _, samples, mean) in calls.items():
            window_mean = inputs.returns_dict["mean"]
            window = calls[(WINDOW_ARM, seed, horizon)][2]
            if arm == INDEPENDENT_ARM:
                np.testing.assert_array_equal(
                    samples, permute_assets(window, independence_rng(seed, horizon))
                )
                np.testing.assert_array_equal(mean, window_mean)
            if arm == WINDOW_ARM:
                np.testing.assert_array_equal(mean, window_mean)
            if arm == JOINTFM_CPU:
                next_day = calls[(JOINTFM_CPU, seed, 1)][2]
                np.testing.assert_allclose(mean, next_day.mean(axis=0))
        _, metadata = read_results(config.output)
        assert metadata.identity["arms"][INDEPENDENT_ARM] == {"window": WINDOW_ARM}

    def test_mean_reading_scores_each_arms_expected_returns(
        self, tmp_path, prices_path, fake_jointfm
    ):
        fake_jointfm.envelope.n_input = 1000
        # Enough trials for every regime tercile of the report.
        config = _config(tmp_path, prices_path, trials=6)
        portfolios = PortfolioScores(
            config.portfolio_sizes, config.rebalancing.cvar, config.assets
        )
        frame = run_experiment(config, [ScenarioScores(), portfolios])
        wide = frame.pivot_table(
            index=["origin", "horizon"], columns="arm", values="mean_correlation"
        )
        # The control keeps the window's mean and reads the same; JointFM its own.
        np.testing.assert_array_equal(wide[INDEPENDENT_ARM], wide[WINDOW_ARM])
        assert not np.allclose(wide[JOINTFM_CPU], wide[WINDOW_ARM])
        assert wide.abs().le(1.0).all().all()

        write_report(config.output, prices_path, tmp_path / "report")
        summary = pd.read_csv(tmp_path / "report" / "summary.csv")
        reported = summary[summary["metric"] == "mean_correlation"]
        assert set(reported["arm"]) == {WINDOW_ARM, JOINTFM_CPU, INDEPENDENT_ARM}
        assert set(reported["horizon"]) == set(HORIZONS)


class TestAttribution:
    PLANTED = {"expected_returns": 0.3, "scenarios": -0.5}

    def _per_trial(self):
        """Per-trial readings with known parts and trial-to-trial noise in each."""
        rng = np.random.default_rng(13)
        rows = []
        for trial in range(40):
            base = rng.normal()
            scenarios = self.PLANTED["scenarios"] + rng.normal(0.0, 0.05)
            expected = self.PLANTED["expected_returns"] + rng.normal(0.0, 0.05)
            for arm, value in (
                (WINDOW_ARM, base),
                (HISTORICAL_MEAN_ARM, base + scenarios),
                ("jointfm-gpu", base + scenarios + expected),
            ):
                rows.append({"trial": trial, "arm": arm, "score": value})
        return pd.DataFrame(rows).set_index(["trial", "arm"])

    def test_known_difference_splits_into_its_two_parts(self):
        described = {
            WINDOW_ARM: {},
            "jointfm-gpu": {},
            HISTORICAL_MEAN_ARM: {"scenarios": "jointfm-gpu"},
        }
        design = mean_attribution_design(described)
        table = mean_attribution(self._per_trial(), ["score"], design).set_index("part")
        planted = {**self.PLANTED, "total": sum(self.PLANTED.values())}
        for part, value in planted.items():
            row = table.loc[part]
            assert row["mean"] == pytest.approx(value, abs=0.03)
            assert row["ci_low"] < value < row["ci_high"]
            assert row["ci_high"] - row["ci_low"] < 0.1
            assert row["trials"] == 40
        assert table.loc["expected_returns", "mean"] + table.loc[
            "scenarios", "mean"
        ] == pytest.approx(table.loc["total", "mean"])

    def test_design_is_absent_or_complete(self):
        assert mean_attribution_design({WINDOW_ARM: {}, "jointfm-gpu": {}}) is None
        with pytest.raises(ValueError, match="jointfm-gpu"):
            mean_attribution_design(
                {WINDOW_ARM: {}, HISTORICAL_MEAN_ARM: {"scenarios": "jointfm-gpu"}}
            )

    def test_backtests_run_every_arm_and_report_the_split(
        self, tmp_path, prices_path, fake_jointfm, monkeypatch, window_scenario_results
    ):
        fake_jointfm.envelope.n_input = 1000
        config = _config(tmp_path, prices_path)
        frame = run_rebalancing_experiment(config)
        assert set(frame["arm"]) == {
            WINDOW_ARM,
            JOINTFM_CPU,
            HISTORICAL_MEAN_ARM,
            INDEPENDENT_ARM,
        }
        # Both JointFM arms of a trial send its seed, so they draw alike.
        assert len({call["seed"] for call in fake_jointfm.forecast_calls}) == 2
        historical = frame[frame["arm"] == HISTORICAL_MEAN_ARM]
        assert set(historical["device"]) == {"CPU"}
        _, metadata = read_results(config.rebalancing.output)
        described = metadata.identity["arms"][HISTORICAL_MEAN_ARM]
        assert described["scenarios"] == JOINTFM_CPU
        assert described["jointfm_settings"]["expected_return"] == "historical"
        drawn = []
        draw_panel = rebalancing_report._interval_panel

        def record_panel(ax, rows, **kwargs):
            drawn.append(rows)
            draw_panel(ax, rows, **kwargs)

        monkeypatch.setattr(rebalancing_report, "_interval_panel", record_panel)
        files = write_rebalancing_report(
            config.rebalancing.output,
            window_scenario_results(config),
            tmp_path / "report",
        )
        path = tmp_path / "report" / "rebalancing_mean_attribution.csv"
        assert path in files
        table = pd.read_csv(path).set_index(["metric", "part"])
        means = table["mean"]
        wide = frame.set_index(["trial", "arm"])["annualized_volatility"].unstack()
        total = (wide[JOINTFM_CPU] - wide[WINDOW_ARM]).mean()
        assert means["annualized_volatility", "total"] == pytest.approx(total)
        assert means["annualized_volatility", "expected_returns"] + means[
            "annualized_volatility", "scenarios"
        ] == pytest.approx(total)
        names = {file.name for file in files}
        assert {
            "rebalancing_mean_attribution.png",
            "rebalancing_mean_attribution.svg",
        } <= names
        # One panel per reading, each part at the table's mean and interval.
        panels = [rows for rows in drawn if rows[0][0] in ATTRIBUTION_STYLES]
        assert len(panels) == len(METRICS)
        for metric, rows in zip(METRICS, panels):
            assert [row[0] for row in rows] == list(ATTRIBUTION_STYLES)
            scale = METRIC_LABELS[metric][3]
            for style, *values in rows:
                expected = table.loc[
                    (metric, style.name), ["mean", "ci_low", "ci_high"]
                ]
                np.testing.assert_allclose(
                    values, scale * expected.to_numpy(dtype=float)
                )


class TestIndependenceEffects:
    DESCRIBED = {WINDOW_ARM: {}, INDEPENDENT_ARM: {"window": WINDOW_ARM}}

    def test_planted_difference_from_the_window_is_recovered(self):
        rng = np.random.default_rng(13)
        rows = []
        for trial in range(40):
            base = rng.normal()
            removed = 0.4 + rng.normal(0.0, 0.05)
            rows.append({"trial": trial, "arm": WINDOW_ARM, "score": base})
            rows.append(
                {"trial": trial, "arm": INDEPENDENT_ARM, "score": base + removed}
            )
        per_trial = pd.DataFrame(rows).set_index(["trial", "arm"])
        design = independence_design(self.DESCRIBED)
        (row,) = independence_effects(per_trial, ["score"], design).itertuples()
        assert row.effect == INDEPENDENCE_EFFECT
        assert row.mean == pytest.approx(0.4, abs=0.03)
        assert row.ci_low < 0.4 < row.ci_high
        assert row.trials == 40

    def test_breach_raised_in_few_trials_keeps_its_mean(self):
        """Over two origins a trial's breach rate is 0, 0.5, or 1, so the median
        per-trial difference is 0 although removing dependence raised it."""
        raised = set(range(5))
        rows = [
            {
                "trial": trial,
                "arm": arm,
                "breach": 0.5 if arm == INDEPENDENT_ARM and trial in raised else 0.0,
            }
            for trial in range(20)
            for arm in self.DESCRIBED
        ]
        per_trial = pd.DataFrame(rows).set_index(["trial", "arm"])
        design = independence_design(self.DESCRIBED)
        (row,) = independence_effects(per_trial, ["breach"], design).itertuples()
        assert row.mean == pytest.approx(0.5 * len(raised) / 20)
        assert row.ci_low > 0

    def test_design_is_absent_or_complete(self):
        assert independence_design({WINDOW_ARM: {}, "jointfm-gpu": {}}) is None
        with pytest.raises(ValueError, match=WINDOW_ARM):
            independence_design({INDEPENDENT_ARM: {"window": WINDOW_ARM}})


def test_rebalancing_measurement_arms_are_drawn_only_by_their_readers(
    tmp_path, prices_path, fake_jointfm, window_scenario_results, saved_figures
):
    """Every per-arm figure of the rebalancing report names the simulators
    alone, and every measurement arm's reader is drawn."""
    fake_jointfm.envelope.n_input = 1000
    arms = (*COMPARISON_ARMS, *SWAP_DONORS)
    config = _config(
        tmp_path, prices_path, arms=arms, rebalancing_arms=(*arms, HISTORICAL_MEAN_ARM)
    )
    run_rebalancing_experiment(config)
    write_rebalancing_report(
        config.rebalancing.output, window_scenario_results(config), tmp_path / "report"
    )
    readers = {
        **dict.fromkeys(SWAP_DONORS, "rebalancing_swap_effects"),
        HISTORICAL_MEAN_ARM: "rebalancing_mean_attribution",
        INDEPENDENT_ARM: "rebalancing_independence_control",
    }
    per_arm = set(saved_figures) - set(readers.values())
    assert {"rebalancing_differences", "rebalancing_paths"} <= per_arm
    for name in per_arm:
        assert "jointfm" in saved_figures[name], name
        named = [arm for arm in DERIVED_DONORS if arm in saved_figures[name]]
        assert named == [], name
    for arm, name in readers.items():
        assert name in saved_figures, arm
    for arm in (HISTORICAL_MEAN_ARM, INDEPENDENT_ARM):
        assert arm in saved_figures[readers[arm]]
    control = pd.read_csv(tmp_path / "report" / "rebalancing_independence_control.csv")
    assert set(control["metric"]) == set(METRICS)

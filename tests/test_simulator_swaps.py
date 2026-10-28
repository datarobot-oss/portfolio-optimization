# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dependence swap arms of the simulator comparison in experiments/: the
Schaake shuffle, the cells in both comparisons and their reports, with the fake
JointFM client of conftest.py."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from experiments.simulator_comparison.analysis import swap_design, swap_effects
from experiments.simulator_comparison.arms import (
    ARMS_BY_NAME,
    JOINTFM_ARM,
    SWAP_DONORS,
    WINDOW_ARM,
    SwapArm,
    scenario_settings,
)
from experiments.simulator_comparison.config import ExperimentConfig
from experiments.simulator_comparison.plotting import EFFECT_STYLES, SIMULATOR_COLORS
from experiments.simulator_comparison.portfolio_analysis import effect_curves
from experiments.simulator_comparison.rebalancing import (
    backtest_scenarios,
    run_rebalancing_experiment,
)
from experiments.simulator_comparison.rebalancing_report import (
    write_rebalancing_report,
)
from experiments.simulator_comparison.report import write_report
from experiments.simulator_comparison.runner import run_experiment
from experiments.simulator_comparison.scores import (
    PortfolioScores,
    ScenarioScores,
    cell_column,
)
from experiments.simulator_comparison.swaps import (
    schaake_shuffle,
    stretch,
    swap_rng,
)
from experiments.simulator_comparison.table import read_results
from experiments.simulator_comparison.trials import RETURNS_SETTINGS
from portfolio_optimization.cvar_utils import generate_cvar_data
from portfolio_optimization.utils import calculate_returns

ASSETS = 4
WINDOW = 25
NUM_SCEN = 200


def _correlated(rng, rows, rho):
    """Gaussian scenarios whose assets share one factor with correlation ``rho``."""
    factor = rng.normal(size=(rows, 1))
    noise = rng.normal(size=(rows, ASSETS))
    return np.sqrt(rho) * factor + np.sqrt(1 - rho) * noise


def test_stretch_repeats_every_row_equally_often():
    """Each row appears floor or ceil of num_scen / rows times, in order."""
    samples = np.arange(WINDOW * 2).reshape(WINDOW, 2)
    stretched = stretch(samples, NUM_SCEN + 3)
    counts = np.bincount(stretched[:, 0] // 2, minlength=WINDOW)
    assert set(counts) <= {
        (NUM_SCEN + 3) // WINDOW,
        (NUM_SCEN + 3) // WINDOW + 1,
    }
    assert np.all(np.diff(stretched[:, 0]) >= 0)
    np.testing.assert_array_equal(stretch(samples, WINDOW), samples)


def test_stretch_refuses_to_drop_scenarios():
    with pytest.raises(ValueError, match="cannot be stretched"):
        stretch(np.zeros((NUM_SCEN + 1, 2)), NUM_SCEN)


def test_swap_keeps_marginal_values_in_dependence_rank_order():
    """Same-size continuous donors: values are the marginal donor's, ranks the other's."""
    rng = np.random.default_rng(1)
    marginals = rng.standard_t(3, size=(NUM_SCEN, ASSETS)) * [1, 2, 3, 4]
    dependence = _correlated(rng, NUM_SCEN, 0.8)
    swapped = schaake_shuffle(
        marginals, dependence, num_scen=NUM_SCEN, rng=np.random.default_rng(2)
    )
    np.testing.assert_array_equal(np.sort(swapped, axis=0), np.sort(marginals, axis=0))
    np.testing.assert_array_equal(
        rankdata(swapped, axis=0), rankdata(dependence, axis=0)
    )


def test_swap_of_a_stretched_template_stays_inside_each_rows_rank_band():
    """A window template's repeated rows each own a band of ranks per asset.

    The scenarios of one window row get the marginal values whose ranks fall in
    that row's band, so the swap carries the window's rank order to the
    precision of the stretch.
    """
    rng = np.random.default_rng(3)
    window = _correlated(rng, WINDOW, 0.5)
    marginals = rng.normal(size=(NUM_SCEN, ASSETS))
    swapped = schaake_shuffle(
        marginals, window, num_scen=NUM_SCEN, rng=np.random.default_rng(4)
    )
    np.testing.assert_array_equal(np.sort(swapped, axis=0), np.sort(marginals, axis=0))
    template = stretch(window, NUM_SCEN)
    scenario_ranks = rankdata(swapped, axis=0)
    low = rankdata(template, method="min", axis=0)
    high = rankdata(template, method="max", axis=0)
    assert np.all((low <= scenario_ranks) & (scenario_ranks <= high))


def test_ties_of_a_stretched_template_break_independently_per_asset():
    """Within one repeated template row, the assets' rank orders differ."""
    window = np.tile(np.arange(WINDOW, dtype=float)[:, None], (1, ASSETS))
    marginals = np.random.default_rng(5).normal(size=(NUM_SCEN, ASSETS))
    swapped = schaake_shuffle(
        marginals, window, num_scen=NUM_SCEN, rng=np.random.default_rng(6)
    )
    block = stretch(np.arange(WINDOW), NUM_SCEN) == 0
    orders = np.argsort(swapped[block], axis=0)
    assert any(not np.array_equal(orders[:, 0], orders[:, a]) for a in range(1, ASSETS))


def test_swap_of_a_donor_with_itself_returns_it_stretched():
    """The window corner: same donor on both sides returns it, ties and all."""
    rng = np.random.default_rng(7)
    # Rounded returns repeat values within an asset, as real zero returns do.
    window = np.round(_correlated(rng, WINDOW, 0.4), 1)
    swapped = schaake_shuffle(
        window, window, num_scen=NUM_SCEN, rng=np.random.default_rng(8)
    )
    np.testing.assert_array_equal(swapped, stretch(window, NUM_SCEN))


def test_swap_with_identical_seeds_is_reproducible():
    rng = np.random.default_rng(9)
    window = _correlated(rng, WINDOW, 0.3)
    marginals = rng.normal(size=(NUM_SCEN, ASSETS))
    first, second = (
        schaake_shuffle(
            marginals, window, num_scen=NUM_SCEN, rng=np.random.default_rng(10)
        )
        for _ in range(2)
    )
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("marginals", "dependence", "message"),
    [
        (np.zeros(NUM_SCEN), np.zeros((NUM_SCEN, ASSETS)), "scenarios, assets"),
        (np.zeros((NUM_SCEN, ASSETS)), np.zeros((NUM_SCEN, 2)), "assets"),
        (np.full((NUM_SCEN, ASSETS), np.nan), np.zeros((NUM_SCEN, ASSETS)), "finite"),
    ],
)
def test_swap_refuses_mismatched_donors(marginals, dependence, message):
    with pytest.raises(ValueError, match=message):
        schaake_shuffle(
            marginals, dependence, num_scen=NUM_SCEN, rng=np.random.default_rng(0)
        )


SWAP_ARMS = tuple(SWAP_DONORS)
# The donors run first, as configured; the fake service runs on CPU.
RUN_ARMS = (WINDOW_ARM, JOINTFM_ARM, *SWAP_ARMS)
RESOLVED = {WINDOW_ARM: WINDOW_ARM, JOINTFM_ARM: "jointfm-cpu"}
RUN_SCEN = 100
RUN_HORIZONS = (1, 3)


def _run_config(tmp_path, prices_path, arms=RUN_ARMS, **overrides):
    """A small comparison of ``arms`` whose rebalancing runs the same arms."""
    values = dict(
        prices=prices_path,
        output=tmp_path / "results.parquet",
        trials=6,
        origins=3,
        assets=4,
        look_back=64,
        origin_history=64,
        forward=8,
        num_scen=RUN_SCEN,
        seed=5,
        arms=arms,
        portfolio_sizes=(1, 2, 4),
        horizons=RUN_HORIZONS,
        rebalancing=dict(
            output=tmp_path / "rebalancing.parquet",
            scenario_results=tmp_path / "results.parquet",
            arms=arms,
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
        universe_sizes=dict(
            output=tmp_path / "universe_sizes.parquet",
            arms=("no_fit-cpu",),
            sizes=(4,),
        ),
    )
    values.update(overrides)
    return ExperimentConfig(**values)


@pytest.fixture()
def prices_path(tmp_path):
    """Correlated random-walk prices with positive drift, and their cash rates."""
    rows = 300
    rng = np.random.default_rng(12)
    dates = pd.date_range("2019-01-01", periods=rows, freq="B")
    returns = 0.001 + 0.01 * _correlated(rng, rows, 0.4)
    log_prices = np.cumsum(np.hstack([returns, returns[:, :2]]), axis=0)
    frame = pd.DataFrame(
        100.0 * np.exp(log_prices),
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


class RecordingSamples:
    """Records every call's origin seed, horizon, and scenarios, in call order."""

    columns = ("rows",)
    parameters = {}

    def __init__(self):
        self.samples = []

    def __call__(self, inputs, horizon, samples, expected_returns):
        self.samples.append((inputs.seed, horizon, samples))
        return {"rows": float(samples.shape[0])}


def _samples_by_arm(frame, measurement):
    """Each recorded call's scenarios keyed by (arm, seed, horizon)."""
    keys = frame[["arm", "seed", "horizon"]].itertuples(index=False, name=None)
    by_key = {}
    for key, (seed, horizon, samples) in zip(keys, measurement.samples):
        assert key[1:] == (seed, horizon)
        by_key[key] = samples
    return by_key


class TestComparison:
    def test_cells_recombine_their_donors_at_every_horizon(
        self, tmp_path, prices_path, fake_jointfm
    ):
        fake_jointfm.envelope.n_input = 1000
        measurement = RecordingSamples()
        config = _run_config(tmp_path, prices_path, trials=2)
        frame = run_experiment(config, [measurement])

        # The cells reuse the JointFM arm's draws: one request per origin.
        assert len(fake_jointfm.forecast_calls) == 2 * config.origins
        samples = _samples_by_arm(frame, measurement)
        swaps = frame[frame["arm"].isin(SWAP_ARMS)]
        assert (swaps["num_scen"] == RUN_SCEN).all()
        assert set(swaps["fit_type"]) == set(SWAP_ARMS)
        for row in swaps.itertuples():
            marginals, dependence = (
                samples[(RESOLVED[donor], row.seed, row.horizon)]
                for donor in SWAP_DONORS[row.arm]
            )
            expected = schaake_shuffle(
                marginals,
                dependence,
                num_scen=RUN_SCEN,
                rng=swap_rng(row.seed, row.horizon),
            )
            cell = samples[(row.arm, row.seed, row.horizon)]
            np.testing.assert_array_equal(cell, expected)
            np.testing.assert_array_equal(
                np.sort(cell, axis=0), np.sort(stretch(marginals, RUN_SCEN), axis=0)
            )
        for row in swaps[swaps["arm"] == "swap-window-window"].itertuples():
            window = samples[(WINDOW_ARM, row.seed, row.horizon)]
            np.testing.assert_array_equal(
                samples[(row.arm, row.seed, row.horizon)], stretch(window, RUN_SCEN)
            )

        _, metadata = read_results(config.output)
        assert metadata.identity["arms"]["swap-window-jointfm"] == {
            "marginals": WINDOW_ARM,
            "dependence": "jointfm-cpu",
            "num_scen": RUN_SCEN,
        }

    def test_cells_added_later_regenerate_their_donors(
        self, tmp_path, prices_path, fake_jointfm
    ):
        """A session that adds the cells to finished donors draws the same scenarios."""
        fake_jointfm.envelope.n_input = 1000
        together = RecordingSamples()
        run_experiment(
            _run_config(tmp_path / "together", prices_path, trials=1), [together]
        )
        calls = len(fake_jointfm.forecast_calls)

        donors_only = _run_config(
            tmp_path / "later", prices_path, arms=RUN_ARMS[:2], trials=1
        )
        run_experiment(donors_only, [RecordingSamples()])
        later = RecordingSamples()
        frame = run_experiment(
            _run_config(tmp_path / "later", prices_path, trials=1), [later]
        )
        # Donors once with their rows, and once more without for the cells.
        assert len(fake_jointfm.forecast_calls) == calls + 2 * donors_only.origins
        new_rows = frame.iloc[-len(later.samples) :]
        assert set(new_rows["arm"]) == set(SWAP_ARMS)
        first = _samples_by_arm(
            read_results(tmp_path / "together" / "results.parquet")[0], together
        )
        for (seed, horizon, samples), arm in zip(later.samples, new_rows["arm"]):
            np.testing.assert_array_equal(samples, first[(arm, seed, horizon)])

    def test_report_splits_readings_into_effects(
        self, tmp_path, prices_path, fake_jointfm
    ):
        fake_jointfm.envelope.n_input = 1000
        config = _run_config(tmp_path, prices_path)
        run_experiment(
            config,
            [
                ScenarioScores(),
                PortfolioScores(
                    config.portfolio_sizes, config.rebalancing.cvar, config.assets
                ),
            ],
        )
        outputs = write_report(config.output, prices_path, tmp_path / "report")
        effects = pd.read_csv(tmp_path / "report" / "swap_effects.csv")
        assert set(effects["effect"]) == {style.name for style in EFFECT_STYLES}
        assert set(effects["horizon"]) == set(RUN_HORIZONS)
        assert "portfolio_equal_4_breach_05" in set(effects["metric"])
        names = {path.name for path in outputs.files}
        for plot in ("swap_effects_horizons", "swap_effects_portfolio_sizes"):
            assert {f"{plot}.png", f"{plot}.svg"} <= names


def test_swap_cells_need_their_donors(tmp_path, prices_path):
    with pytest.raises(ValueError, match="next to their donors"):
        _run_config(tmp_path, prices_path, arms=(WINDOW_ARM, "swap-jointfm-window"))


def test_every_cell_has_a_figure_color():
    assert set(SWAP_DONORS) <= set(SIMULATOR_COLORS)


class TestEffects:
    DESCRIBED = {
        WINDOW_ARM: {},
        "jointfm-gpu": {},
        "swap-jointfm-window": {"marginals": "jointfm-gpu"},
        "swap-window-jointfm": {"marginals": WINDOW_ARM},
        "swap-window-window": {"marginals": WINDOW_ARM},
    }

    def _per_trial(self, effects):
        """Per-trial readings whose cells add up the planted ``effects``."""
        rng = np.random.default_rng(13)
        rows = []
        design = {
            "jointfm-gpu": (1, 1),
            "swap-jointfm-window": (1, 0),
            "swap-window-jointfm": (0, 1),
            "swap-window-window": (0, 0),
        }
        for trial in range(40):
            base = rng.normal()
            noise = rng.normal(0.0, 1e-3)
            rows.append({"trial": trial, "arm": WINDOW_ARM, "score": base})
            for arm, (marginal, dependence) in design.items():
                value = (
                    base
                    + effects["marginals"] * marginal
                    + effects["dependence"] * dependence
                    + effects["interaction"] * marginal * dependence
                    + effects["construction"]
                    + noise * (arm == "jointfm-gpu")
                )
                rows.append({"trial": trial, "arm": arm, "score": value})
        return pd.DataFrame(rows).set_index(["trial", "arm"])

    def test_planted_effects_are_recovered(self):
        planted = {
            "marginals": 0.3,
            "dependence": -0.5,
            "interaction": 0.2,
            "construction": 0.05,
        }
        per_trial = self._per_trial(planted)
        table = swap_effects(per_trial, ["score"], swap_design(self.DESCRIBED))
        means = table.set_index("effect")["mean"]
        # Each main effect averages over the other factor, so it carries half the interaction.
        expected = {
            "marginals": 0.3 + 0.1,
            "dependence": -0.5 + 0.1,
            "interaction": 0.2,
            "construction": 0.05,
        }
        for effect, value in expected.items():
            assert means[effect] == pytest.approx(value, abs=2e-3)
        row = table.set_index("effect").loc["dependence"]
        assert row["ci_low"] <= row["mean"] <= row["ci_high"]
        cells = per_trial["score"].unstack("arm")
        corner = (cells["jointfm-gpu"] - cells["swap-window-window"]).mean()
        assert means["marginals"] + means["dependence"] == pytest.approx(corner)

    def test_effect_in_few_trials_is_kept(self):
        """A breach rate that JointFM's dependence raises in a quarter of the trials.

        Over two origins a trial's breach rate is 0, 0.5, or 1, so the median
        per-trial effect is 0 although the effect is there.
        """
        raised = set(range(5))
        jointfm_dependence = ("jointfm-gpu", "swap-window-jointfm")
        rows = [
            {
                "trial": trial,
                "arm": arm,
                "breach": 0.5 if trial in raised and arm in jointfm_dependence else 0.0,
            }
            for trial in range(20)
            for arm in self.DESCRIBED
        ]
        per_trial = pd.DataFrame(rows).set_index(["trial", "arm"])
        table = swap_effects(per_trial, ["breach"], swap_design(self.DESCRIBED))
        dependence = table.set_index("effect").loc["dependence"]
        assert dependence["mean"] == pytest.approx(0.5 * len(raised) / 20)
        assert dependence["ci_low"] > 0

    def test_portfolio_effects_become_size_curves(self):
        """Each portfolio cell's effect lands at its kind, size, and reading."""
        cells = [("equal", 1), ("equal", 4), ("long_short", 4)]
        rows = [
            {"trial": trial, "arm": arm, "score": 0.0}
            | {
                cell_column(kind, size, reading): float(trial * size)
                * (arm == "jointfm-gpu")
                for kind, size in cells
                for reading in ("breach_05", "crps")
            }
            for trial in range(4)
            for arm in self.DESCRIBED
        ]
        per_trial = pd.DataFrame(rows).set_index(["trial", "arm"])
        columns = [c for c in per_trial.columns if c.startswith("portfolio_")]
        effects = swap_effects(
            per_trial, ["score", *columns], swap_design(self.DESCRIBED)
        )
        curves = effect_curves(effects)
        # The scenario score is no portfolio cell, so it draws no size curve.
        assert len(curves) == len(columns) * len(EFFECT_STYLES)
        assert set(curves["arm"]) == {style.name for style in EFFECT_STYLES}
        row = curves[
            (curves["kind"] == "long_short")
            & (curves["size"] == 4)
            & (curves["reading"] == "crps")
            & (curves["arm"] == "interaction")
        ].iloc[0]
        # Only the JointFM corner moves, by trial * size: a mean of 1.5 * 4.
        assert row["estimate"] == pytest.approx(6.0)
        assert row["ci_low"] <= row["estimate"] <= row["ci_high"]

    def test_design_is_absent_or_complete(self):
        assert swap_design({WINDOW_ARM: {}, "jointfm-gpu": {}}) is None
        with pytest.raises(ValueError, match="needs all of"):
            swap_design(
                {"swap-window-window": {"marginals": WINDOW_ARM}, WINDOW_ARM: {}}
            )
        with pytest.raises(ValueError, match="donors"):
            swap_design(
                {
                    "swap-jointfm-window": {"marginals": "jointfm-gpu"},
                    "swap-window-jointfm": {"marginals": WINDOW_ARM},
                    "swap-window-window": {"marginals": WINDOW_ARM},
                    WINDOW_ARM: {},
                }
            )


class TestRebalancing:
    def test_generator_keeps_the_marginal_donors_values_and_mean(
        self, prices_path, fake_jointfm
    ):
        fake_jointfm.envelope.n_input = 1000
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        returns_dict = calculate_returns(
            prices.iloc[:65, :ASSETS], returns_compute_settings=RETURNS_SETTINGS
        )
        jointfm = ARMS_BY_NAME["jointfm-cpu"]
        window = ARMS_BY_NAME[WINDOW_ARM]
        arm = SwapArm("swap-jointfm-window", jointfm, window)
        settings, generator = backtest_scenarios(arm, num_scen=RUN_SCEN, seed=3)
        assert settings == scenario_settings(jointfm, num_scen=RUN_SCEN, seed=3)
        cvar_data = generator(dict(returns_dict), settings, cash_return=0.001)[
            "cvar_data"
        ]
        donor = generate_cvar_data(dict(returns_dict), settings)["cvar_data"]
        template = generate_cvar_data(
            dict(returns_dict), scenario_settings(window, num_scen=RUN_SCEN, seed=3)
        )["cvar_data"]
        np.testing.assert_array_equal(cvar_data.mean, donor.mean)
        np.testing.assert_array_equal(
            cvar_data.R.T,
            schaake_shuffle(
                donor.R.T, template.R.T, num_scen=RUN_SCEN, rng=swap_rng(3, 1)
            ),
        )
        np.testing.assert_allclose(cvar_data.p, 1 / RUN_SCEN)
        assert cvar_data.cash_return == 0.001

    def test_backtests_run_every_cell_and_report_the_effects(
        self, tmp_path, prices_path, fake_jointfm, window_scenario_results
    ):
        fake_jointfm.envelope.n_input = 1000
        config = _run_config(tmp_path, prices_path)
        frame = run_rebalancing_experiment(config)
        assert set(frame["arm"]) == {WINDOW_ARM, "jointfm-cpu", *SWAP_ARMS}
        per_trial = frame.groupby("trial")
        assert (per_trial["risk_aversion"].nunique() == 1).all()
        # Every JointFM request of a trial, the cells' included, carries its seed.
        assert len({call["seed"] for call in fake_jointfm.forecast_calls}) == 2
        files = write_rebalancing_report(
            config.rebalancing.output,
            window_scenario_results(config),
            tmp_path / "report",
        )
        effects = pd.read_csv(tmp_path / "report" / "rebalancing_swap_effects.csv")
        assert set(effects["effect"]) == {style.name for style in EFFECT_STYLES}
        names = {path.name for path in files}
        assert {"rebalancing_swap_effects.png", "rebalancing_swap_effects.svg"} <= names


def test_swap_cells_need_scenarios_for_every_window_row(tmp_path, prices_path):
    with pytest.raises(ValueError, match="at least look_back 64"):
        _run_config(tmp_path, prices_path, num_scen=50)

# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired run comparison: two results files that differ in one stated setting.

Two configurations that differ in one setting and their output files draw the
same trials, origins, and seeds, so each arm's readings pair trial by trial
across their results files. The comparison states the setting (``SETTINGS``),
refuses two files whose identities differ in anything else, and reads, per arm
both files hold and per reading, the change from the baseline file to the
other one:

- ``look_back``: the history every arm receives, on scenario results. Every
  arm gets it, so each arm's correlation bias still reads against the window
  it saw.
- ``horizons``: the horizons each origin is scored at, on scenario results,
  paired at the horizons both files hold. JointFM answers them in one request,
  so its next day can change with the others; every other arm's cannot.
- ``risk_aversion``: the base risk aversion, on rebalancing results.
- ``checkpoint``: the checkpoint the JointFM services serve, on either
  results. A service's label does not identify the weights behind it, so two
  files that record the same service are compared too.

A reading that is a share of events, a VaR breach rate or an interval
coverage, is compared by its mean change over trials, and every other reading
by its median change, as the comparisons table is: over a few origins per
trial a share takes few values, so its median change reads zero where its
mean keeps the change. Run it as
``python -m experiments.simulator_comparison.paired``.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    BOOTSTRAP_SEED,
    COMPARED_SCORES,
    MIN_TRIALS,
    _wilcoxon_p_value,
    aggregate,
    bootstrap_interval,
    complete_trials,
)
from .config import load_config
from .plotting import (
    INK,
    INK_SECONDARY,
    ArmStyle,
    _arm_legend,
    _curves,
    _figure,
    _interval_panel,
    _save,
    _target_line,
    _title,
    arm_styles,
)
from .portfolio_report import KIND_LABELS
from .portfolios import KINDS
from .rebalancing import METRICS
from .rebalancing_report import METRIC_LABELS, complete_runs, per_trial_frame
from .report import SCORE_LABELS
from .scores import PORTFOLIO_READINGS, cell_column, recorded_cells
from .table import identity_differences, read_results

SCENARIO = "scenario"
REBALANCING = "rebalancing"
RESULTS = (SCENARIO, REBALANCING)
# Identity entries the readings do not depend on.
IGNORED_IDENTITY_KEYS = ("host",)
# Readings that are shares of events, compared by their mean change.
RATE_SCORES = ("cover_50", "cover_90", "ew_var_exceeded", "chosen_breach_rate")
RATE_CELL_READINGS = tuple(r for r in PORTFOLIO_READINGS if r.startswith("breach_"))
# The PIT bins are a histogram, read in the reports, not a reading to pair.
CELL_READINGS = tuple(r for r in PORTFOLIO_READINGS if not r.startswith("pit_"))
STATISTICS = {"mean": np.mean, "median": np.median}
# Readings printed after a comparison; its table holds every reading.
HEADLINES = {
    SCENARIO: ("corr_bias_trailing", "corr_error_forward", "ew_var_exceeded", "crps"),
    REBALANCING: (
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "chosen_breach_rate",
    ),
}
# (title, unit) of the portfolio readings drawn over the portfolio size.
PORTFOLIO_FIGURE_READINGS = {
    "breach_05": ("95 % VaR breach rate", "share of portfolio-days"),
    "breach_01": ("99 % VaR breach rate", "share of portfolio-days"),
    "crps": ("Portfolio CRPS", "portfolio trailing volatility × √h units"),
}
SCENARIO_PANEL_COLUMNS = 3
REBALANCING_PANEL_COLUMNS = 4
# Leading columns of every table row: what the two files differ in.
SETTING_COLUMNS = ("setting", "baseline_value", "value")


def _lookup(identity: dict, path: Sequence[str]):
    value = identity
    for key in path:
        value = value[key]
    return value


def _replace(identity: dict, path: Sequence[str], value) -> dict:
    """A copy of ``identity`` with ``value`` at ``path``; other entries are shared."""
    head, *rest = path
    return {**identity, head: _replace(identity[head], rest, value) if rest else value}


@dataclass(frozen=True)
class IdentitySetting:
    """A setting stored at one place of a results file's identity.

    Attributes
    ----------
    name : str
        The name a comparison states.
    path : tuple of str
        Identity keys down to the setting's value.
    results : tuple of str
        The results, of ``RESULTS``, whose readings the setting changes.
    """

    name: str
    path: tuple[str, ...]
    results: tuple[str, ...]

    def value(self, identity: dict) -> str:
        """The setting's value in ``identity``, as the table records it."""
        return json.dumps(_lookup(identity, self.path))

    def align(self, other: dict, baseline: dict) -> dict:
        """``other`` with the baseline's value of the setting."""
        return _replace(other, self.path, _lookup(baseline, self.path))

    def check_pairs(self, baseline: dict, other: dict) -> None:
        """Refuse two identities that hold the same value of the setting.

        Raises
        ------
        ValueError
            If both identities hold one value.
        """
        if self.value(baseline) == self.value(other):
            raise ValueError(
                f"Both files hold {self.name} {self.value(baseline)}; a paired "
                f"comparison of {self.name} needs two values."
            )


@dataclass(frozen=True)
class CheckpointSetting:
    """The checkpoint the JointFM services serve, which may keep its label.

    A service records the label it was started with, which stays the same when
    the weights behind it change, so two files that record the same services
    are compared too: only the person who served them knows whether the
    weights differ. A service record also holds the runner's image and the
    client's version, which change with the service.
    """

    name: str = "checkpoint"
    results: tuple[str, ...] = RESULTS

    def value(self, identity: dict) -> str:
        """The model and checkpoint labels of the identity's JointFM services."""
        return ", ".join(
            sorted(
                {
                    f"{service['model_version']} {service['checkpoint_version']}"
                    for service in identity["services"].values()
                }
            )
        )

    def align(self, other: dict, baseline: dict) -> dict:
        """``other`` with the baseline's record of every service both hold."""
        services = {
            arm: baseline["services"].get(arm, service)
            for arm, service in other["services"].items()
        }
        return {**other, "services": services}

    def check_pairs(self, baseline: dict, other: dict) -> None:
        """Refuse two identities that share no JointFM arm.

        Raises
        ------
        ValueError
            If no JointFM arm is in both identities, so no reading the
            checkpoint changes pairs.
        """
        if not baseline["services"].keys() & other["services"].keys():
            raise ValueError(
                f"The files share no JointFM arm (baseline "
                f"{sorted(baseline['services'])}, other {sorted(other['services'])}), "
                f"so no reading of the checkpoint pairs; serve both checkpoints on "
                f"one device."
            )


Setting = IdentitySetting | CheckpointSetting
SETTINGS: dict[str, Setting] = {
    setting.name: setting
    for setting in (
        IdentitySetting("look_back", ("experiment", "look_back"), (SCENARIO,)),
        IdentitySetting("horizons", ("experiment", "horizons"), (SCENARIO,)),
        IdentitySetting(
            "risk_aversion", ("rebalancing", "cvar", "risk_aversion"), (REBALANCING,)
        ),
        CheckpointSetting(),
    )
}


def results_kind(identity: dict) -> str:
    """Whether an identity is a scenario or a rebalancing results file's."""
    return REBALANCING if "rebalancing" in identity else SCENARIO


def check_comparable(
    setting: Setting,
    results: str,
    baseline_path: Path,
    baseline: dict,
    other_path: Path,
    other: dict,
) -> None:
    """Refuse two results identities that differ in more than ``setting``.

    Arms may differ, as in a resumed file; an arm both hold must keep its
    settings and, unless ``setting`` is the checkpoint, its service.

    Raises
    ------
    ValueError
        If ``setting`` does not change ``results``, either file holds other
        results, the setting does not pair (``check_pairs``), or anything
        besides the setting, the arms each holds, and
        ``IGNORED_IDENTITY_KEYS`` differs.
    """
    if results not in setting.results:
        raise ValueError(
            f"{setting.name} changes the {' and '.join(setting.results)} results, "
            f"not the {results} results."
        )
    for path, identity in ((baseline_path, baseline), (other_path, other)):
        if results_kind(identity) != results:
            raise ValueError(
                f"{path} holds {results_kind(identity)} results, not {results} results."
            )
    setting.check_pairs(baseline, other)
    aligned = {
        **setting.align(other, baseline),
        **{key: baseline[key] for key in IGNORED_IDENTITY_KEYS},
    }
    differing = identity_differences(baseline, aligned)
    if differing:
        details = "; ".join(
            f"{key}: baseline {old!r}, other {new!r}"
            for key, (old, new) in differing.items()
        )
        raise ValueError(
            f"{baseline_path} and {other_path} differ in more than {setting.name}, "
            f"so their readings would mix both changes. {details}"
        )


def paired_change(baseline: pd.Series, other: pd.Series, statistic: str) -> dict:
    """Change of one arm's reading from the baseline run to the other one.

    Parameters
    ----------
    baseline, other : pd.Series
        The arm's per-trial readings in each run, indexed by trial; a trial
        that either lacks or holds as NaN is left out.
    statistic : str
        ``mean`` or ``median``, the summary over trials of ``STATISTICS``.

    Returns
    -------
    dict
        ``statistic``; ``difference``, the statistic over trials of other
        minus baseline, with its bootstrap interval; the Wilcoxon p-value;
        and the number of paired trials.

    Raises
    ------
    ValueError
        If fewer than ``MIN_TRIALS`` trials pair.
    """
    paired = pd.concat(
        {"baseline": baseline, "other": other}, axis=1, join="inner"
    ).dropna()
    if len(paired) < MIN_TRIALS:
        raise ValueError(
            f"{len(paired)} trials pair; a paired comparison needs at least "
            f"{MIN_TRIALS}."
        )
    differences = (paired["other"] - paired["baseline"]).to_numpy(dtype=float)
    summarize = STATISTICS[statistic]
    low, high = bootstrap_interval(
        differences, summarize, np.random.default_rng(BOOTSTRAP_SEED)
    )
    return {
        "statistic": statistic,
        "difference": float(summarize(differences)),
        "ci_low": low,
        "ci_high": high,
        "p_value": _wilcoxon_p_value(differences),
        "trials": len(differences),
    }


def _changes(
    baseline: pd.DataFrame,
    other: pd.DataFrame,
    arms: Sequence[str],
    columns: Sequence[str],
    rates: set[str],
) -> list[dict]:
    """Every reading's change per arm between two (trial, arm) tables."""
    return [
        {
            "arm": arm,
            "metric": column,
            **paired_change(
                baseline[column].xs(arm, level="arm"),
                other[column].xs(arm, level="arm"),
                "mean" if column in rates else "median",
            ),
        }
        for column in columns
        for arm in arms
    ]


def scenario_changes(
    runs: Sequence[tuple[pd.DataFrame, dict]], arms: Sequence[str]
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Every score and portfolio reading's change per horizon both runs hold.

    Each run keeps the trials in which ``arms`` have every origin at every one
    of its own horizons.

    Parameters
    ----------
    runs : sequence of (pd.DataFrame, dict)
        The baseline's and the other run's rows and identity.
    arms : sequence of str
        Arms both runs hold.

    Returns
    -------
    tuple
        One row per horizon, reading, and arm, with the change of
        ``paired_change``; and the portfolio cells the readings cover.
    """
    horizons = sorted(
        set(runs[0][1]["experiment"]["horizons"])
        & set(runs[1][1]["experiment"]["horizons"])
    )
    per_trial = []
    for frame, identity in runs:
        experiment = identity["experiment"]
        complete, _ = complete_trials(
            frame[frame["arm"].isin(arms)],
            tuple(arms),
            experiment["origins"],
            len(experiment["horizons"]),
        )
        per_trial.append(
            {h: aggregate(complete[complete["horizon"] == h]) for h in horizons}
        )
    cells = recorded_cells(runs[0][0].columns)
    columns = [
        *COMPARED_SCORES,
        *(
            cell_column(kind, size, reading)
            for kind, size in cells
            for reading in CELL_READINGS
        ),
    ]
    rates = {
        *RATE_SCORES,
        *(
            cell_column(kind, size, reading)
            for kind, size in cells
            for reading in RATE_CELL_READINGS
        ),
    }
    rows = [
        {"horizon": horizon, **row}
        for horizon in horizons
        for row in _changes(
            per_trial[0][horizon], per_trial[1][horizon], arms, columns, rates
        )
    ]
    return pd.DataFrame(rows), cells


def rebalancing_changes(
    runs: Sequence[tuple[pd.DataFrame, dict]], arms: Sequence[str]
) -> pd.DataFrame:
    """Every rebalancing reading's change per arm, over the trials both completed.

    Each run keeps the trials in which every arm of ``arms`` finished.
    """
    baseline, other = (
        per_trial_frame(complete_runs(frame, tuple(arms))) for frame, _ in runs
    )
    return pd.DataFrame(_changes(baseline, other, arms, METRICS, set(RATE_SCORES)))


@dataclass(frozen=True)
class PairedComparison:
    """The change of every reading between two results files.

    Attributes
    ----------
    setting : str
        The setting the files differ in, a key of ``SETTINGS``.
    results : str
        Which results the files hold, of ``RESULTS``.
    baseline_value, value : str
        The setting's value in the baseline and in the other file.
    table : pd.DataFrame
        One row per arm and reading, and per horizon for scenario results:
        ``SETTING_COLUMNS``, ``horizon`` for scenario results, ``arm``,
        ``metric``, and the change of ``paired_change``.
    cells : list of (str, int)
        The (kind, size) portfolio cells of scenario results; empty for
        rebalancing results.
    styles : list of ArmStyle
        How each arm both files hold is drawn.
    """

    setting: str
    results: str
    baseline_value: str
    value: str
    table: pd.DataFrame
    cells: list[tuple[str, int]]
    styles: list[ArmStyle]


def paired_comparison(
    baseline_path: Path, other_path: Path, setting_name: str, results: str
) -> PairedComparison:
    """Paired per-trial change of every reading from one results file to another.

    Parameters
    ----------
    baseline_path, other_path : Path
        Results files measured under the same conditions except the setting.
    setting_name : str
        The setting they differ in, a key of ``SETTINGS``.
    results : str
        Which results both files hold, of ``RESULTS``.

    Raises
    ------
    ValueError
        If the setting is unknown, the files are not comparable
        (``check_comparable``), share no arm, or either holds fewer complete
        trials, or both fewer paired trials, than ``MIN_TRIALS``.
    """
    if setting_name not in SETTINGS:
        raise ValueError(
            f"Unknown setting {setting_name!r}; choose from {list(SETTINGS)}."
        )
    setting = SETTINGS[setting_name]
    runs = [
        (frame, metadata.identity)
        for frame, metadata in map(read_results, (baseline_path, other_path))
    ]
    (baseline_frame, baseline), (_, other) = runs
    check_comparable(setting, results, baseline_path, baseline, other_path, other)
    arms = [arm for arm in baseline["arms"] if arm in other["arms"]]
    if not arms:
        raise ValueError(f"{baseline_path} and {other_path} share no arm.")
    if results == SCENARIO:
        changes, cells = scenario_changes(runs, arms)
    else:
        changes, cells = rebalancing_changes(runs, arms), []
    baseline_value, value = setting.value(baseline), setting.value(other)
    table = changes.assign(
        setting=setting.name, baseline_value=baseline_value, value=value
    )[[*SETTING_COLUMNS, *changes.columns]]
    return PairedComparison(
        setting=setting.name,
        results=results,
        baseline_value=baseline_value,
        value=value,
        table=table,
        cells=cells,
        styles=arm_styles(baseline_frame[baseline_frame["arm"].isin(arms)]),
    )


def _suptitle(comparison: PairedComparison) -> str:
    return (
        f"Paired change from {comparison.setting} {comparison.baseline_value} to "
        f"{comparison.value}: median over trials, or mean for shares of events, "
        f"with its 95 % bootstrap interval; 0 is no change"
    )


def _estimates(changes: pd.DataFrame) -> pd.DataFrame:
    """Changes in the columns ``plotting._curves`` reads."""
    return changes.rename(columns={"difference": "estimate"})


def plot_scenario_changes(comparison: PairedComparison, directory: Path) -> list[Path]:
    """One panel per compared score: each arm's change over the horizons."""
    table = comparison.table
    rows = -(-len(COMPARED_SCORES) // SCENARIO_PANEL_COLUMNS)
    figure, axes = _figure(14, 3.6 * rows, rows=rows, cols=SCENARIO_PANEL_COLUMNS)
    for ax in list(axes.flat)[len(COMPARED_SCORES) :]:
        ax.set_visible(False)
    for ax, column in zip(axes.flat, COMPARED_SCORES):
        title, unit, _ = SCORE_LABELS[column]
        changes = table[table["metric"] == column]
        _target_line(ax, 0.0)
        _curves(ax, _estimates(changes), "horizon", comparison.styles)
        ax.set_xlabel("trading days after the origin", fontsize=8)
        _title(ax, title, f"{changes['statistic'].iloc[0]} change, {unit}")
    figure.suptitle(_suptitle(comparison), color=INK, fontsize=10)
    _arm_legend(figure, comparison.styles)
    return _save(figure, directory, f"paired_{comparison.setting}")


def plot_portfolio_changes(comparison: PairedComparison, directory: Path) -> list[Path]:
    """Each arm's change of the next-day portfolio readings over the portfolio size.

    Rows are the readings of ``PORTFOLIO_FIGURE_READINGS``, columns the
    portfolio kinds of the results, as in the report's portfolio-size figure.
    """
    next_day = comparison.table[comparison.table["horizon"] == 1]
    recorded = {kind for kind, _ in comparison.cells}
    kinds = [kind for kind in KINDS if kind in recorded]
    readings = PORTFOLIO_FIGURE_READINGS
    figure, axes = _figure(
        5.5 * len(kinds), 3.6 * len(readings), rows=len(readings), cols=len(kinds)
    )
    for row, (reading, (title, unit)) in enumerate(readings.items()):
        for column, kind in enumerate(kinds):
            ax = axes[row, column]
            sizes = {
                cell_column(kind, size, reading): size
                for cell_kind, size in comparison.cells
                if cell_kind == kind
            }
            changes = next_day[next_day["metric"].isin(sizes)]
            _target_line(ax, 0.0)
            _curves(
                ax,
                _estimates(changes).assign(size=changes["metric"].map(sizes)),
                "size",
                comparison.styles,
            )
            ax.set_xlabel("assets per portfolio", fontsize=8)
            _title(
                ax,
                f"{title} · {KIND_LABELS[kind]}",
                f"{changes['statistic'].iloc[0]} change at the next day, {unit}",
            )
    figure.suptitle(_suptitle(comparison), color=INK, fontsize=10)
    _arm_legend(figure, comparison.styles)
    return _save(figure, directory, f"paired_{comparison.setting}_portfolios")


def plot_rebalancing_changes(
    comparison: PairedComparison, directory: Path
) -> list[Path]:
    """One panel per rebalancing reading: each arm's change with its interval."""
    table = comparison.table
    rows = -(-len(METRICS) // REBALANCING_PANEL_COLUMNS)
    figure, axes = _figure(
        16, 2.6 * rows + 0.8, rows=rows, cols=REBALANCING_PANEL_COLUMNS
    )
    for ax in list(axes.flat)[len(METRICS) :]:
        ax.set_visible(False)
    for ax, metric in zip(axes.flat, METRICS):
        title, unit, _, scale = METRIC_LABELS[metric]
        changes = table[table["metric"] == metric].set_index("arm")
        _interval_panel(
            ax,
            [
                (
                    style,
                    *(
                        scale
                        * changes.loc[
                            style.name, ["difference", "ci_low", "ci_high"]
                        ].to_numpy(dtype=float)
                    ),
                )
                for style in comparison.styles
            ],
            reference=0.0,
            reference_label="no change",
        )
        ax.set_xlabel(unit, color=INK_SECONDARY, fontsize=8)
        _title(ax, title, f"{changes['statistic'].iloc[0]} change")
    figure.suptitle(_suptitle(comparison), color=INK, fontsize=10)
    _arm_legend(figure, comparison.styles)
    return _save(figure, directory, f"paired_{comparison.setting}")


def write_paired_comparison(
    comparison: PairedComparison, output_dir: Path
) -> list[Path]:
    """Write the comparison's table and figures to ``output_dir``.

    Files are named after the setting: ``paired_<setting>.csv`` and the
    figure ``paired_<setting>``, and for scenario results also the portfolio
    figure ``paired_<setting>_portfolios``, each as PNG and SVG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"paired_{comparison.setting}.csv"
    comparison.table.to_csv(path, index=False)
    if comparison.results == SCENARIO:
        figures = [
            *plot_scenario_changes(comparison, output_dir),
            *plot_portfolio_changes(comparison, output_dir),
        ]
    else:
        figures = plot_rebalancing_changes(comparison, output_dir)
    return [path, *figures]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.simulator_comparison.paired",
        description="Paired change of every reading between the results of two "
        "simulator-comparison configurations that differ in one setting and their "
        "output files, written as paired_<setting>.csv and figures into the "
        "--config report directory of the compared results.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="YAML configuration of the reference run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration of the other run; the outputs go to its report.",
    )
    parser.add_argument(
        "--setting",
        required=True,
        choices=list(SETTINGS),
        help="The one setting the two runs differ in.",
    )
    parser.add_argument(
        "--results",
        required=True,
        choices=list(RESULTS),
        help="Which results file of each configuration to compare: the scenario "
        "comparison's output or the rebalancing backtest's rebalancing.output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline = load_config(args.baseline)
    config = load_config(args.config)
    locations = {
        SCENARIO: (baseline.output, config.output, config.report_dir),
        REBALANCING: (
            baseline.rebalancing.output,
            config.rebalancing.output,
            config.rebalancing.report_dir,
        ),
    }
    baseline_path, other_path, output_dir = locations[args.results]
    comparison = paired_comparison(
        baseline_path, other_path, args.setting, args.results
    )
    files = write_paired_comparison(comparison, output_dir)
    table = comparison.table
    headline = table[table["metric"].isin(HEADLINES[args.results])].drop(
        columns=list(SETTING_COLUMNS)
    )
    print(f"{comparison.setting}: {comparison.baseline_value} -> {comparison.value}")
    with pd.option_context("display.width", 160, "display.precision", 4):
        print(headline.to_string(index=False))
    print(f"\nPaired comparison finished: {len(files)} files in {output_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

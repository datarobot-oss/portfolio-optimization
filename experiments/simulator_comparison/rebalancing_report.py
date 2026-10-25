# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Report of a rebalancing results file: tables and plots.

A trial counts when every arm finished its run, and every comparison pairs
arms within a trial against ``no_fit``, the historical scenarios, with
bootstrap intervals over trials, as the scenario report does. With the
dependence swap arms it also splits every reading into what JointFM's
marginals and its dependence cause (``rebalancing_swap_effects.csv``, from
``analysis.swap_effects``), and with the historical-mean JointFM arm into
what JointFM's expected returns and its scenarios cause
(``rebalancing_mean_attribution.csv``, from ``analysis.mean_attribution``).
Whether the backtest sees dependence at all reads off the independence
control's row in ``rebalancing_comparisons.csv``. How the risk of the
portfolios each arm chose compares with their outcome, against the binomial
band of a correct VaR, is ``rebalancing_chosen_risk.csv`` (``chosen_risk``).
Colors, fills, and labels follow that report.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import binom

from .analysis import (
    BOOTSTRAP_SEED,
    CONFIDENCE,
    MIN_TRIALS,
    REFERENCE_ARM,
    bootstrap_interval,
    compare,
    mean_attribution,
    mean_attribution_design,
    swap_design,
    swap_effects,
)
from .chosen import CHOSEN_METRICS
from .config import load_config
from .plotting import (
    INK,
    INK_SECONDARY,
    LINE_WIDTH,
    MARKER_EDGE,
    SURFACE,
    ArmStyle,
    _device_legend,
    _figure,
    _interval_panel,
    _save,
    _target_line,
    _title,
    arm_styles,
)
from .rebalancing import METRICS
from .table import read_results

PERCENT = 100
# (title, unit of the difference, which direction is better or None, scale to
# display units) per reading.
METRIC_LABELS = {
    "mean_step_return": (
        "Mean step return",
        "percentage points per step",
        "higher",
        PERCENT,
    ),
    "step_return_std": (
        "Step-return volatility",
        "percentage points per step",
        "lower",
        PERCENT,
    ),
    "total_return": (
        "Total return",
        "percentage points over the window",
        "higher",
        PERCENT,
    ),
    "annualized_return": (
        "Annualized return",
        "percentage points per year",
        "higher",
        PERCENT,
    ),
    "annualized_volatility": (
        "Annualized volatility",
        "percentage points per year",
        "lower",
        PERCENT,
    ),
    "sharpe_ratio": ("Sharpe ratio", "annualized", "higher", 1),
    "sortino_ratio": ("Sortino ratio", "annualized, downside risk only", "higher", 1),
    "max_drawdown": (
        "Maximum drawdown",
        "percentage points from the running peak",
        "lower",
        PERCENT,
    ),
    "calmar_ratio": (
        "Calmar ratio",
        "annualized return over max drawdown",
        "higher",
        1,
    ),
    "turnover": ("Turnover", "sum of absolute weight changes", None, 1),
    "rebalances": ("Rebalances", "re-optimizations triggered", None, 1),
    "buy_and_hold_return": (
        "Buy-and-hold return",
        "percentage points, first portfolio kept",
        "higher",
        PERCENT,
    ),
    "chosen_breach_rate": (
        "Chosen-portfolio VaR breaches",
        "percentage points of held days",
        None,
        PERCENT,
    ),
    "chosen_volatility_ratio": (
        "Chosen-portfolio volatility ratio",
        "realized over scenario volatility",
        None,
        1,
    ),
    "chosen_effective_holdings": (
        "Effective holdings",
        "assets, (sum |w|)^2 / sum w^2",
        None,
        1,
    ),
}
# Which side of the zero line a better arm lands on, per better direction.
BETTER_SIDE = {"higher": "right is better", "lower": "left is better"}


def complete_runs(frame: pd.DataFrame, arms: tuple[str, ...]) -> pd.DataFrame:
    """Rows of the trials in which every arm finished.

    Raises
    ------
    ValueError
        If fewer than ``MIN_TRIALS`` trials are complete.
    """
    frame = frame[frame["arm"].isin(arms)]
    counts = frame.groupby("trial")["arm"].nunique()
    complete = counts.index[counts == len(arms)]
    if len(complete) < MIN_TRIALS:
        raise ValueError(
            f"{len(complete)} trials have every arm's run; the report needs at "
            f"least {MIN_TRIALS}."
        )
    return frame[frame["trial"].isin(complete)]


def per_trial_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Readings indexed by (trial, arm), with each arm's simulator and device."""
    return frame.set_index(["trial", "arm"])[[*METRICS, "fit_type", "device"]]


def comparisons(per_trial: pd.DataFrame) -> pd.DataFrame:
    """Every reading of every arm against ``REFERENCE_ARM``, paired by trial."""
    arms = per_trial.index.get_level_values("arm").unique()
    if REFERENCE_ARM not in arms:
        raise ValueError(
            f"The rebalancing report compares every arm with {REFERENCE_ARM}."
        )
    return pd.DataFrame(
        [
            compare(per_trial, metric, arm, REFERENCE_ARM)
            for metric in METRICS
            for arm in arms
            if arm != REFERENCE_ARM
        ]
    )


def summary(per_trial: pd.DataFrame) -> pd.DataFrame:
    """Per arm and reading: mean, median, and bootstrap interval of the mean."""
    rows = []
    for metric in METRICS:
        wide = per_trial[metric].unstack("arm")
        for arm in wide.columns:
            values = wide[arm].to_numpy(dtype=float)
            low, high = bootstrap_interval(
                values, np.mean, np.random.default_rng(BOOTSTRAP_SEED)
            )
            rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "ci_low": low,
                    "ci_high": high,
                    "trials": len(values),
                }
            )
    return pd.DataFrame(rows)


def _root_mean_square(values: np.ndarray, axis: int) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=axis))


def chosen_risk(frame: pd.DataFrame, confidence: float) -> pd.DataFrame:
    """Per arm: how the risk of the portfolios it chose compares with their outcome.

    Every run holds its portfolios over the same number of trading days, so
    the mean of the runs' breach rates is the share of all held days whose
    realized return fell below the chosen portfolio's scenario VaR, and the
    root mean square of the runs' volatility ratios is the ratio over all
    held days. Both come with bootstrap intervals over trials, and the breach
    rate with the binomial band, the range of shares that a VaR stating the
    risk correctly gives over that many independent days at ``CONFIDENCE``.
    The band ignores that every arm is read on the same days; ``no_fit``'s
    row, which also picks the portfolios its scenarios understate most, is the
    reference to read the others against, and ``rebalancing_comparisons.csv``
    pairs them by trial.

    Raises
    ------
    ValueError
        If a run has no chosen-portfolio record, as a run recorded before it
        existed and resumed into the file, or the runs hold their portfolios
        over different numbers of days.
    """
    if frame[["chosen_days", *CHOSEN_METRICS]].isna().any(axis=None):
        raise ValueError(
            "Some runs have no chosen-portfolio record; the file mixes runs "
            "recorded before it existed. Rerun the rebalancing from scratch."
        )
    days = frame["chosen_days"].map(np.sum)
    if days.nunique() != 1:
        raise ValueError(
            f"The runs hold their portfolios over {sorted(days.unique())} days; "
            f"the chosen-portfolio readings need one count."
        )
    level = 1 - confidence
    rows = []
    for arm, runs in frame.groupby("arm", sort=False):
        total = int(days.loc[runs.index].sum())
        breaches = runs["chosen_breach_rate"].to_numpy(dtype=float)
        ratios = runs["chosen_volatility_ratio"].to_numpy(dtype=float)
        holdings = runs["chosen_effective_holdings"].to_numpy(dtype=float)
        breach_low, breach_high = bootstrap_interval(
            breaches, np.mean, np.random.default_rng(BOOTSTRAP_SEED)
        )
        ratio_low, ratio_high = bootstrap_interval(
            ratios, _root_mean_square, np.random.default_rng(BOOTSTRAP_SEED)
        )
        band_low, band_high = binom.interval(CONFIDENCE, total, level)
        rows.append(
            {
                "arm": arm,
                "trials": len(runs),
                "days": total,
                "var_level": level,
                "breach_rate": float(breaches.mean()),
                "breach_ci_low": breach_low,
                "breach_ci_high": breach_high,
                "breach_band_low": float(band_low / total),
                "breach_band_high": float(band_high / total),
                "volatility_ratio": float(_root_mean_square(ratios, axis=0)),
                "volatility_ratio_ci_low": ratio_low,
                "volatility_ratio_ci_high": ratio_high,
                "effective_holdings": float(holdings.mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_differences(table: pd.DataFrame, styles: Sequence[ArmStyle], directory: Path):
    """One panel per reading: each arm's median paired difference from the reference.

    Pairing by trial removes the spread between trials (universe and period)
    that every arm shares, so the panels show only what the simulator changes.
    """
    contenders = [style for style in styles if style.name != REFERENCE_ARM]
    cols = 4
    rows = -(-len(METRICS) // cols)
    figure, axes = _figure(16, 2.6 * rows + 0.8, rows=rows, cols=cols)
    for ax in list(axes.flat)[len(METRICS) :]:
        ax.set_visible(False)
    for ax, metric in zip(axes.flat, METRICS):
        title, unit, better, scale = METRIC_LABELS[metric]
        readings = table[table["metric"] == metric].set_index("arm")
        _interval_panel(
            ax,
            [
                (
                    style,
                    *(
                        scale
                        * readings.loc[
                            style.name, ["median_difference", "ci_low", "ci_high"]
                        ].to_numpy(dtype=float)
                    ),
                )
                for style in contenders
            ],
            reference=0.0,
            reference_label=REFERENCE_ARM,
        )
        ax.set_xlabel(unit, color=INK_SECONDARY, fontsize=8)
        side = f"; {BETTER_SIDE[better]}" if better else ""
        _title(ax, title, f"minus {REFERENCE_ARM}, paired by trial{side}")
    figure.suptitle(
        f"Rebalancing readings against {REFERENCE_ARM}: median paired difference "
        f"and 95 % bootstrap interval over trials",
        color=INK,
        fontsize=10,
    )
    _device_legend(figure)
    return _save(figure, directory, "rebalancing_differences")


def plot_paths(frame: pd.DataFrame, styles: Sequence[ArmStyle], directory: Path):
    """Mean cumulative return after each step over trials, per arm."""
    figure, axes = _figure(9, 5.5)
    ax = axes[0, 0]
    ax.grid(axis="x", visible=False)
    for style in styles:
        returns = np.array(list(frame.loc[frame["arm"] == style.name, "step_returns"]))
        paths = np.cumprod(1 + returns, axis=1) - 1
        mean = np.concatenate([[0.0], paths.mean(axis=0)]) * PERCENT
        steps = np.arange(len(mean))
        ax.plot(
            steps,
            mean,
            color=style.color,
            linewidth=LINE_WIDTH,
            linestyle="-" if style.filled else "--",
            marker="o",
            markersize=5,
            markerfacecolor=style.color if style.filled else SURFACE,
            markeredgecolor=SURFACE if style.filled else style.color,
            markeredgewidth=MARKER_EDGE,
            label=style.inline_label,
            zorder=3,
        )
        ax.text(
            steps[-1] + 0.3,
            mean[-1],
            f"{style.inline_label} {mean[-1]:+.1f} %",
            va="center",
            color=INK_SECONDARY,
            fontsize=8,
        )
    _target_line(ax, 0)
    ax.set_xlim(0, steps[-1] + 4)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("step", color=INK_SECONDARY, fontsize=8)
    ax.set_ylabel("mean cumulative return (%)", color=INK_SECONDARY, fontsize=8)
    _title(
        ax,
        "Rebalancing paths",
        "mean over trials of the cumulative return after each step",
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper left")
    return _save(figure, directory, "rebalancing_paths")


def write_rebalancing_report(results_path: Path, output_dir: Path) -> list[Path]:
    """Compute the comparisons of a rebalancing results file and write its outputs.

    Raises
    ------
    ValueError
        If fewer trials than the report needs have every arm's run, the
        reference arm is missing, or the swap design or the expected-return
        attribution is incomplete.
    """
    frame, metadata = read_results(results_path)
    frame = complete_runs(frame, tuple(metadata.identity["arms"]))
    per_trial = per_trial_frame(frame)
    styles = arm_styles(per_trial)
    table = comparisons(per_trial)
    tables = {
        "rebalancing_summary": summary(per_trial),
        "rebalancing_comparisons": table,
        "rebalancing_chosen_risk": chosen_risk(
            frame, metadata.identity["rebalancing"]["cvar"]["confidence"]
        ),
    }
    design = swap_design(metadata.identity["arms"])
    if design is not None:
        tables["rebalancing_swap_effects"] = swap_effects(per_trial, METRICS, design)
    attribution = mean_attribution_design(metadata.identity["arms"])
    if attribution is not None:
        tables["rebalancing_mean_attribution"] = mean_attribution(
            per_trial, METRICS, attribution
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name, data in tables.items():
        path = output_dir / f"{name}.csv"
        data.to_csv(path, index=False)
        files.append(path)
    files += plot_differences(table, styles, output_dir)
    files += plot_paths(frame, styles, output_dir)
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.simulator_comparison.rebalancing_report",
        description="Tables and plots of a configuration's rebalancing results, "
        "written to <rebalancing output stem>_report beside them; runs nothing.",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="YAML configuration file."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    rebalancing = load_config(build_parser().parse_args(argv).config).rebalancing
    outputs = write_rebalancing_report(rebalancing.output, rebalancing.report_dir)
    print(
        f"Rebalancing report finished: {len(outputs)} files in "
        f"{rebalancing.report_dir}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

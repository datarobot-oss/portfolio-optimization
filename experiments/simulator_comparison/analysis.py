# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Statistics of a results table: per-trial aggregates and paired comparisons.

A trial is the unit of replication: its arms saw the same assets, windows, and
seeds, so every comparison pairs arms within a trial, and confidence intervals
resample trials. Trials share assets and dates, so the intervals measure
sensitivity to the universe and the period, not sampling error of a population.
Every statistic reads one horizon's rows; the report computes it per horizon
and stacks the results (``stack_horizons``).
"""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .arms import HISTORICAL_MEAN_ARM
from .scores import (
    PIT_BINS,
    PORTFOLIO_CRPS,
    PORTFOLIO_READINGS,
    ScenarioScores,
    cell_column,
    recorded_cells,
)

REFERENCE_ARM = "no_fit-cpu"
# Proper scores the arms are ranked by, each on its own: the energy score of
# the asset vector, which barely reacts to wrong correlations, and two that do.
RANKING_SCORES = ("energy_score", "variogram_score", PORTFOLIO_CRPS)
SKILL_SCORES = (*RANKING_SCORES, "crps")
PIT_COLUMNS = tuple(c for c in ScenarioScores.columns if c.startswith("pit_bin_"))
COMPARED_SCORES = (
    *(c for c in ScenarioScores.columns if c not in PIT_COLUMNS),
    PORTFOLIO_CRPS,
)
# Aggregated over a trial's origins by the median instead of the mean.
MEDIAN_SCORES = ("spread",)
RELIABILITY_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 2000
# Fixed so a report re-renders identically from the same table.
BOOTSTRAP_SEED = 0
MIN_TRIALS = 2


def complete_trials(
    frame: pd.DataFrame, arms: tuple[str, ...], n_origins: int, n_horizons: int
) -> tuple[pd.DataFrame, list[int]]:
    """Keep the trials in which every arm has every origin at every horizon.

    Returns
    -------
    tuple
        The rows of complete trials and the indices of the trials left out.

    Raises
    ------
    ValueError
        If fewer than ``MIN_TRIALS`` trials are complete.
    """
    counts = frame.groupby("trial").size()
    complete = counts.index[counts == len(arms) * n_origins * n_horizons]
    excluded = sorted(int(t) for t in counts.index.difference(complete))
    if len(complete) < MIN_TRIALS:
        raise ValueError(
            f"{len(complete)} complete trials; the report needs at least "
            f"{MIN_TRIALS}. A trial is complete when every arm has every origin; "
            f"leave out an arm whose session is still running (report --exclude, "
            f"task experiments:report EXCLUDE=<arm>)."
        )
    return frame[frame["trial"].isin(complete)], excluded


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (trial, arm): scores over the trial's origins, and timing.

    Every score, PIT bin, and portfolio-cell reading is the mean over the
    trial's origins, except ``MEDIAN_SCORES``. ``origins`` counts the origins
    aggregated. ``seconds`` is the mean over the trial's non-warm-up calls; it
    is NaN for a (trial, arm) whose only calls were warm-ups.

    Raises
    ------
    ValueError
        If ``frame`` holds rows of more than one horizon, whose scores would
        mix in one mean.
    """
    horizons = sorted(frame["horizon"].unique())
    if len(horizons) != 1:
        raise ValueError(f"aggregate reads one horizon's rows; got {horizons}.")
    cell_columns = [
        cell_column(kind, size, reading)
        for kind, size in recorded_cells(frame.columns)
        for reading in PORTFOLIO_READINGS
    ]
    grouped = frame.groupby(["trial", "arm"])
    means = grouped[[*COMPARED_SCORES, *PIT_COLUMNS, *cell_columns]].mean()
    for column in MEDIAN_SCORES:
        means[column] = grouped[column].median()
    timed = frame[~frame["warmup"]].groupby(["trial", "arm"])["seconds"].mean()
    # One concat instead of column inserts, which fragment a frame this wide.
    means = pd.concat(
        [
            means,
            grouped.size().rename("origins"),
            timed.reindex(means.index).rename("seconds"),
        ],
        axis=1,
    )
    arms = frame.groupby("arm")[["fit_type", "device"]].first()
    return means.join(arms, on="arm")


def bootstrap_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile interval of ``statistic`` over trials resampled with replacement.

    ``statistic`` reduces along ``axis`` like ``np.mean`` and ``np.median``,
    so all resamples are evaluated in one call.
    """
    resamples = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    estimates = statistic(values[resamples], axis=1)
    tail = (1 - CONFIDENCE) / 2
    low, high = np.quantile(estimates, [tail, 1 - tail])
    return float(low), float(high)


def _wilcoxon_p_value(differences: np.ndarray) -> float:
    # Identical arms give all-zero differences, where the test is undefined
    # and there is no evidence of a difference.
    if not differences.any():
        return 1.0
    return float(wilcoxon(differences).pvalue)


def _wide(per_trial: pd.DataFrame, column: str) -> pd.DataFrame:
    return per_trial[column].unstack("arm")


def compare(per_trial: pd.DataFrame, column: str, arm: str, reference: str) -> dict:
    """Median over trials of ``arm`` minus ``reference``, with interval and p-value."""
    wide = _wide(per_trial, column)[[arm, reference]].dropna()
    differences = (wide[arm] - wide[reference]).to_numpy()
    low, high = bootstrap_interval(
        differences, np.median, np.random.default_rng(BOOTSTRAP_SEED)
    )
    return {
        "metric": column,
        "arm": arm,
        "reference": reference,
        "median_difference": float(np.median(differences)),
        "ci_low": low,
        "ci_high": high,
        "p_value": _wilcoxon_p_value(differences),
        "trials": len(differences),
    }


def device_pairs(per_trial: pd.DataFrame) -> list[tuple[str, str]]:
    """(GPU arm, CPU arm) of every simulator that ran on both devices."""
    arms = per_trial.groupby("arm")[["fit_type", "device"]].first()
    pairs = []
    for fit_type, group in arms.groupby("fit_type"):
        by_device = dict(zip(group["device"], group.index))
        if {"CPU", "GPU"} <= set(by_device):
            pairs.append((by_device["GPU"], by_device["CPU"]))
    return pairs


def comparisons(per_trial: pd.DataFrame) -> pd.DataFrame:
    """Every score of every arm against ``REFERENCE_ARM``, and GPU against CPU.

    Raises
    ------
    ValueError
        If the table has no ``REFERENCE_ARM``.
    """
    arms = per_trial.index.get_level_values("arm").unique()
    if REFERENCE_ARM not in arms:
        raise ValueError(
            f"The report compares every arm with {REFERENCE_ARM}, which the "
            f"results do not contain."
        )
    rows = [
        compare(per_trial, column, arm, REFERENCE_ARM)
        for column in COMPARED_SCORES
        for arm in arms
        if arm != REFERENCE_ARM
    ]
    rows += [
        compare(per_trial, column, gpu, cpu)
        for column in COMPARED_SCORES
        for gpu, cpu in device_pairs(per_trial)
    ]
    return pd.DataFrame(rows)


def skill(per_trial: pd.DataFrame) -> pd.DataFrame:
    """Percent reduction of each trial's score against ``REFERENCE_ARM``'s.

    Positive skill means the arm's scenarios made the realized days more
    typical than the look-back window's own returns did.
    """
    frames = []
    for column in SKILL_SCORES:
        wide = _wide(per_trial, column)
        relative = 100 * (1 - wide.div(wide[REFERENCE_ARM], axis=0))
        long = relative.drop(columns=REFERENCE_ARM).stack().rename("skill")
        frames.append(long.reset_index().assign(score=column))
    return pd.concat(frames, ignore_index=True)


def skill_summary(skills: pd.DataFrame) -> pd.DataFrame:
    """Median skill per score and arm with its bootstrap interval."""
    rows = []
    for (score, arm), group in skills.groupby(["score", "arm"], sort=False):
        values = group["skill"].to_numpy()
        low, high = bootstrap_interval(
            values, np.median, np.random.default_rng(BOOTSTRAP_SEED)
        )
        rows.append(
            {
                "score": score,
                "arm": arm,
                "median_skill": float(np.median(values)),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def ranking(per_trial: pd.DataFrame) -> pd.DataFrame:
    """Per score of ``RANKING_SCORES`` and arm: mean rank and share of trials won.

    Each score ranks the arms on its own (1 is best), and the rankings are
    never merged, since any weights between them would be arbitrary. Tied arms
    share the better rank, and each counts the trial as won.

    Returns
    -------
    pd.DataFrame
        Columns ``score``, ``arm``, ``mean_rank``, ``share_won``; per score
        in ``RANKING_SCORES`` order, best mean rank first.
    """
    tables = []
    for score in RANKING_SCORES:
        ranks = _wide(per_trial, score).rank(axis=1, method="min")
        table = pd.DataFrame(
            {"mean_rank": ranks.mean(), "share_won": (ranks == 1).mean()}
        ).sort_values("mean_rank")
        tables.append(table.rename_axis("arm").reset_index().assign(score=score))
    return pd.concat(tables, ignore_index=True)[
        ["score", "arm", "mean_rank", "share_won"]
    ]


def timing(per_trial: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean seconds per call per arm, and each simulator's GPU speedup.

    The speedup is the per-trial ratio of the CPU arm's mean seconds to the
    GPU arm's; both come with bootstrap intervals over trials.
    """
    seconds = _wide(per_trial, "seconds")
    rows = []
    for arm in seconds.columns:
        values = seconds[arm].dropna().to_numpy()
        low, high = bootstrap_interval(
            values, np.mean, np.random.default_rng(BOOTSTRAP_SEED)
        )
        rows.append(
            {
                "arm": arm,
                "mean_seconds": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
                "trials": len(values),
            }
        )
    speedups = []
    for gpu, cpu in device_pairs(per_trial):
        ratios = (seconds[cpu] / seconds[gpu]).dropna().to_numpy()
        low, high = bootstrap_interval(
            ratios, np.median, np.random.default_rng(BOOTSTRAP_SEED)
        )
        speedups.append(
            {
                "gpu_arm": gpu,
                "cpu_arm": cpu,
                "median_speedup": float(np.median(ratios)),
                "ci_low": low,
                "ci_high": high,
                "trials": len(ratios),
            }
        )
    speedup_columns = [
        "gpu_arm",
        "cpu_arm",
        "median_speedup",
        "ci_low",
        "ci_high",
        "trials",
    ]
    return (
        pd.DataFrame(rows).set_index("arm"),
        pd.DataFrame(speedups, columns=speedup_columns),
    )


def reliability(per_trial: pd.DataFrame) -> pd.DataFrame:
    """Empirical coverage of central intervals per arm, read off the PIT bins.

    A central ``level`` interval holds the outcome when its PIT lies within
    ``[(1 - level) / 2, (1 + level) / 2]``; every level in
    ``RELIABILITY_LEVELS`` falls on bin edges, so the coverage is a sum of bins.
    """
    pit = per_trial[list(PIT_COLUMNS)].groupby("arm").mean()
    coverage = {}
    for level in RELIABILITY_LEVELS:
        low = (1 - level) / 2 * PIT_BINS
        high = (1 + level) / 2 * PIT_BINS
        if not (np.isclose(low, round(low)) and np.isclose(high, round(high))):
            raise ValueError(f"Level {level} does not fall on the PIT bin edges.")
        coverage[level] = pit.iloc[:, round(low) : round(high)].sum(axis=1)
    return pd.DataFrame(coverage)


def summary(per_trial: pd.DataFrame) -> pd.DataFrame:
    """Per arm and score: mean, median, and bootstrap interval of the mean."""
    rows = []
    for column in (*COMPARED_SCORES, "seconds"):
        wide = _wide(per_trial, column)
        for arm in wide.columns:
            values = wide[arm].dropna().to_numpy()
            low, high = bootstrap_interval(
                values, np.mean, np.random.default_rng(BOOTSTRAP_SEED)
            )
            rows.append(
                {
                    "arm": arm,
                    "metric": column,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "ci_low": low,
                    "ci_high": high,
                    "trials": len(values),
                }
            )
    return pd.DataFrame(rows)


def swap_design(described_arms: Mapping[str, dict]) -> dict[str, str] | None:
    """The arm of every cell of the dependence swap design, or None without it.

    ``described_arms`` is a results file's ``arms`` identity (``describe_arms``),
    which names each swap cell's resolved donors, so the JointFM corner is the
    JointFM arm the cells drew from. Cells are keyed by their marginal source,
    then their dependence source: ``J`` JointFM, ``W`` the window.

    Returns
    -------
    dict or None
        ``JJ``, ``JW``, ``WJ``, ``WW``, and ``window`` (``REFERENCE_ARM``) to arm
        names when all three swap cells are among ``described_arms``; None
        when none is.

    Raises
    ------
    ValueError
        If only some of the swap cells are present, or a donor is not, which
        leaves the design incomplete.
    """
    cells = {
        "JW": "swap-jointfm-window",
        "WJ": "swap-window-jointfm",
        "WW": "swap-window-window",
    }
    present = [name for name in cells.values() if name in described_arms]
    if not present:
        return None
    if len(present) != len(cells):
        raise ValueError(
            f"The swap design needs all of {list(cells.values())}; the results "
            f"hold {present}."
        )
    jointfm = described_arms[cells["JW"]]["marginals"]
    missing = [arm for arm in (jointfm, REFERENCE_ARM) if arm not in described_arms]
    if missing:
        raise ValueError(f"The swap design needs its donors {missing} as well.")
    return {"JJ": jointfm, **cells, "window": REFERENCE_ARM}


def swap_effects(
    per_trial: pd.DataFrame, columns: Sequence[str], design: Mapping[str, str]
) -> pd.DataFrame:
    """Main effects and interaction of the dependence swap design per reading.

    Per trial and reading, with cells named marginal source first:

    - ``marginals``: ((JJ - WJ) + (JW - WW)) / 2, JointFM's marginals against
      the window's, averaged over both dependence sources;
    - ``dependence``: ((JJ - JW) + (WJ - WW)) / 2, JointFM's dependence
      against the window's, averaged over both marginal sources;
    - ``interaction``: (JJ - WJ) - (JW - WW), how much JointFM's marginals
      change the reading more under JointFM's dependence than the window's;
    - ``construction``: WW - window, what stretching and tie-breaking alone
      change, zero within noise when the swap adds nothing of its own.

    Returns
    -------
    pd.DataFrame
        One row per reading and effect: median over trials, its bootstrap
        interval, the Wilcoxon p-value, and the trial count.
    """
    rows = []
    for column in columns:
        wide = _wide(per_trial, column)[list(design.values())].dropna()
        cell = {key: wide[arm].to_numpy() for key, arm in design.items()}
        effects = {
            "marginals": (cell["JJ"] - cell["WJ"] + cell["JW"] - cell["WW"]) / 2,
            "dependence": (cell["JJ"] - cell["JW"] + cell["WJ"] - cell["WW"]) / 2,
            "interaction": cell["JJ"] - cell["WJ"] - cell["JW"] + cell["WW"],
            "construction": cell["WW"] - cell["window"],
        }
        for effect, values in effects.items():
            low, high = bootstrap_interval(
                values, np.median, np.random.default_rng(BOOTSTRAP_SEED)
            )
            rows.append(
                {
                    "metric": column,
                    "effect": effect,
                    "median": float(np.median(values)),
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": _wilcoxon_p_value(values),
                    "trials": len(values),
                }
            )
    return pd.DataFrame(rows)


def mean_attribution_design(
    described_arms: Mapping[str, dict],
) -> dict[str, str] | None:
    """The arms of the expected-return attribution, or None without it.

    ``described_arms`` is a results file's ``arms`` identity (``describe_arms``),
    which names the JointFM arm the historical-mean arm drew its scenarios from.

    Returns
    -------
    dict or None
        ``jointfm``, ``historical``, and ``window`` (``REFERENCE_ARM``) to arm
        names when ``HISTORICAL_MEAN_ARM`` is among ``described_arms``; None
        when it is not.

    Raises
    ------
    ValueError
        If the historical-mean arm's JointFM donor or the reference arm is
        missing, which leaves the split incomplete.
    """
    if HISTORICAL_MEAN_ARM not in described_arms:
        return None
    jointfm = described_arms[HISTORICAL_MEAN_ARM]["scenarios"]
    missing = [arm for arm in (jointfm, REFERENCE_ARM) if arm not in described_arms]
    if missing:
        raise ValueError(
            f"The expected-return attribution needs {missing} next to "
            f"{HISTORICAL_MEAN_ARM}."
        )
    return {
        "jointfm": jointfm,
        "historical": HISTORICAL_MEAN_ARM,
        "window": REFERENCE_ARM,
    }


def mean_attribution(
    per_trial: pd.DataFrame, columns: Sequence[str], design: Mapping[str, str]
) -> pd.DataFrame:
    """Split each reading's JointFM-minus-window difference into its two causes.

    Per trial and reading, with ``design`` from ``mean_attribution_design``:

    - ``total``: jointfm - window, the difference the comparisons table reads;
    - ``expected_returns``: jointfm - historical, the same scenarios on
      JointFM's expected returns instead of the window's mean;
    - ``scenarios``: historical - window, JointFM's scenarios instead of the
      window's on the same expected returns.

    The two parts add up to the total in every trial. Each is summarized by
    its mean over trials, not the median of the comparisons table, because
    means add up too: the two parts' means sum to the total's.

    Returns
    -------
    pd.DataFrame
        One row per reading and part: mean over trials, its bootstrap
        interval, the Wilcoxon p-value, and the trial count.
    """
    rows = []
    for column in columns:
        wide = _wide(per_trial, column)[list(design.values())].dropna()
        arm = {key: wide[name].to_numpy() for key, name in design.items()}
        parts = {
            "total": arm["jointfm"] - arm["window"],
            "expected_returns": arm["jointfm"] - arm["historical"],
            "scenarios": arm["historical"] - arm["window"],
        }
        for part, values in parts.items():
            low, high = bootstrap_interval(
                values, np.mean, np.random.default_rng(BOOTSTRAP_SEED)
            )
            rows.append(
                {
                    "metric": column,
                    "part": part,
                    "mean": float(values.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": _wilcoxon_p_value(values),
                    "trials": len(values),
                }
            )
    return pd.DataFrame(rows)


def stack_horizons(tables: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """One table of per-horizon tables, with a leading ``horizon`` column."""
    stacked = pd.concat(
        [table.assign(horizon=horizon) for horizon, table in tables.items()],
        ignore_index=True,
    )
    return stacked[["horizon", *(c for c in stacked.columns if c != "horizon")]]

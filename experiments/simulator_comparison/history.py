# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""History comparison: the same trials measured at two look-back lengths.

Two configurations that differ only in ``look_back`` and their results files
draw the same trials, origins, and scenario seeds, because the trial draw
reads ``origin_history`` alone. Pairing their results trial by trial reads
what a longer history changes for every arm. Every arm gets the same history
in each run, so each arm's correlation bias still reads it against the window
it saw. Run it as ``python -m experiments.simulator_comparison.history``.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .analysis import COMPARED_SCORES, aggregate, compare, complete_trials
from .config import load_config
from .table import merge_identity, read_results

HISTORY_FILE = "history_comparison.csv"
# Identity entries the compared scores do not depend on.
IGNORED_IDENTITY_KEYS = ("host",)
# Readings printed after a comparison; the file holds every compared score.
HEADLINE_SCORES = (
    "corr_bias_trailing",
    "corr_error_forward",
    "ew_var_exceeded",
    "crps",
)


def check_comparable(
    baseline_path: Path, baseline: dict, other_path: Path, other: dict
) -> None:
    """Refuse two results identities that differ in more than ``look_back``.

    Arms may differ, as in a resumed file; an arm both hold must keep its
    settings and service, so both runs read one checkpoint.

    Raises
    ------
    ValueError
        If both files hold the same ``look_back``, or if anything besides it,
        the arms' presence, and ``IGNORED_IDENTITY_KEYS`` differs.
    """
    look_back = baseline["experiment"]["look_back"]
    if other["experiment"]["look_back"] == look_back:
        raise ValueError(
            f"{baseline_path} and {other_path} both hold look_back {look_back}; "
            f"a history comparison needs two lengths."
        )
    aligned = {
        **other,
        "experiment": {**other["experiment"], "look_back": look_back},
        **{key: baseline[key] for key in IGNORED_IDENTITY_KEYS},
    }
    merge_identity(other_path, baseline, aligned)


def history_comparison(baseline_path: Path, other_path: Path) -> pd.DataFrame:
    """Paired per-trial change of every compared score from one history to another.

    Each run keeps the trials in which its arms have every origin at every
    horizon, and a score pairs the trials both runs completed.

    Parameters
    ----------
    baseline_path, other_path : Path
        Results files measured under the same conditions except ``look_back``.

    Returns
    -------
    pd.DataFrame
        One row per horizon, arm both files hold, and score of
        ``COMPARED_SCORES``: ``look_back`` and ``baseline_look_back``, and the
        median over trials of other minus baseline with its bootstrap
        interval, Wilcoxon p-value, and number of paired trials.

    Raises
    ------
    ValueError
        If the files are not comparable (``check_comparable``), share no arm,
        or either holds fewer complete trials than the report needs.
    """
    baseline_frame, baseline = read_results(baseline_path)
    other_frame, other = read_results(other_path)
    check_comparable(baseline_path, baseline.identity, other_path, other.identity)
    arms = tuple(
        arm for arm in baseline.identity["arms"] if arm in other.identity["arms"]
    )
    if not arms:
        raise ValueError(f"{baseline_path} and {other_path} share no arm.")
    experiment = baseline.identity["experiment"]
    horizons = experiment["horizons"]
    runs = {}
    for identity, frame in (
        (baseline.identity, baseline_frame),
        (other.identity, other_frame),
    ):
        complete, _ = complete_trials(
            frame[frame["arm"].isin(arms)], arms, experiment["origins"], len(horizons)
        )
        runs[identity["experiment"]["look_back"]] = complete
    base_look_back = experiment["look_back"]
    other_look_back = other.identity["experiment"]["look_back"]
    rows = []
    for horizon in horizons:
        combined = pd.concat(
            aggregate(frame[frame["horizon"] == horizon]).rename(
                index=lambda arm, look_back=look_back: f"{arm}@{look_back}", level="arm"
            )
            for look_back, frame in runs.items()
        )
        for column in COMPARED_SCORES:
            for arm in arms:
                paired = compare(
                    combined,
                    column,
                    f"{arm}@{other_look_back}",
                    f"{arm}@{base_look_back}",
                )
                rows.append(
                    {
                        "horizon": horizon,
                        "arm": arm,
                        "metric": column,
                        "look_back": other_look_back,
                        "baseline_look_back": base_look_back,
                        **{
                            key: paired[key]
                            for key in (
                                "median_difference",
                                "ci_low",
                                "ci_high",
                                "p_value",
                                "trials",
                            )
                        },
                    }
                )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.simulator_comparison.history",
        description="Paired change of every score between two simulator-comparison "
        "configurations that differ only in look_back, written to "
        f"<output stem>_report/{HISTORY_FILE} beside the --config results.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="YAML configuration of the reference history.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML configuration of the other history; the table goes to its report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline = load_config(args.baseline)
    config = load_config(args.config)
    table = history_comparison(baseline.output, config.output)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    path = config.report_dir / HISTORY_FILE
    table.to_csv(path, index=False)
    with pd.option_context("display.width", 160, "display.precision", 4):
        print(table[table["metric"].isin(HEADLINE_SCORES)].to_string(index=False))
    print(f"\nHistory comparison finished: {path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebalancing backtest of nested asset subsets of each trial's universe.

Each trial fixes one random order of its assets (``asset_order``), and a
universe of size n trades the first n of them (``sized_trial``), so every
smaller universe is a subset of every larger one and the size equal to
``assets`` trades the rebalancing backtest's universe. Every size and arm of a
trial runs over the trial's trading window with its scenario seed and the
rebalancing settings through ``rebalancing.run_backtest``, so the readings of
a trial differ across sizes only in the assets traded and across arms only in
the simulator. Each size calibrates its own risk aversion on its own assets
(``rebalancing.calibrated_cvar_params``), as a trial of that size would.

The blog reads the results (``complete_universe_runs``) to draw each
simulator's total return over the sizes and to read each simulator's change
in total return and maximum drawdown from the smallest to the largest size
(``blog.universe_size_table``).
"""

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from portfolio_optimization import risk_free
from portfolio_optimization.gpu import apply_gpu_selection

from .analysis import MIN_TRIALS
from .arms import check_same_checkpoint, resolve_arms
from .config import ExperimentConfig, load_config
from .rebalancing import (
    calibrated_cvar_params,
    draw_rebalancing_trial,
    rebalancing_identity,
    verified_backtest,
    write_universe,
)
from .runner import PROGRESS_INTERVAL_LOG, PROGRESS_INTERVAL_TERMINAL, load_prices
from .table import RunMetadata, merge_identity, read_results, write_results
from .trials import Trial

KEY_COLUMNS = ("trial", "size", "arm")
# Separates a trial's asset order from its other draws, so adding it changed
# none of the trials, windows, or seeds already drawn.
UNIVERSE_SIZE_STREAM = 0x53495A45


def asset_order(trial: Trial, seed: int) -> tuple[str, ...]:
    """The trial's assets in the random order whose first n form its size-n universe."""
    rng = np.random.default_rng([seed, trial.index, UNIVERSE_SIZE_STREAM])
    return tuple(trial.assets[i] for i in rng.permutation(len(trial.assets)))


def sized_trial(trial: Trial, order: Sequence[str], size: int) -> Trial:
    """The trial on the first ``size`` assets of ``order``, kept in the table's column order.

    Raises
    ------
    ValueError
        If ``order`` is not an order of the trial's assets or ``size`` does not
        fit it.
    """
    if sorted(order) != sorted(trial.assets):
        raise ValueError(f"The asset order is not one of trial {trial.index}'s assets.")
    if not 1 <= size <= len(order):
        raise ValueError(
            f"A universe of {size} assets does not fit trial {trial.index}'s "
            f"{len(order)}."
        )
    chosen = set(order[:size])
    return dataclasses.replace(
        trial, assets=tuple(asset for asset in trial.assets if asset in chosen)
    )


def run_universe_size_experiment(
    config: ExperimentConfig, command: Sequence[str] = ()
) -> pd.DataFrame:
    """Run every pending (trial, size, arm) backtest and append it to the results file.

    Rows already in ``config.universe_sizes.output`` are skipped, so a run
    resumes, and more trials or sizes extend it; a file measured under other
    conditions is refused, as for the rebalancing backtest.

    Raises
    ------
    ValueError
        If the existing file was measured under another identity, a JointFM
        service serves another checkpoint, or a run does not cover its steps.
    """
    rebalancing = config.rebalancing
    sweep = config.universe_sizes
    prices = load_prices(config.prices)
    rates = risk_free.load_risk_free_rates(rebalancing.risk_free_rates)
    arms, services = resolve_arms(sweep.arms)
    identity = rebalancing_identity(config, arms, prices, services)
    frame = None
    commands = [list(command)]
    if sweep.output.exists():
        frame, stored = read_results(sweep.output)
        identity = merge_identity(sweep.output, stored.identity, identity)
        commands = stored.commands + commands
    check_same_checkpoint(identity["services"])
    metadata = RunMetadata(identity=identity, commands=commands)
    done = (
        set()
        if frame is None
        else set(frame[list(KEY_COLUMNS)].itertuples(index=False, name=None))
    )
    work = [
        (index, size, arm)
        for index in range(rebalancing.trials)
        for size in sweep.sizes
        for arm in arms
        if (index, size, arm.name) not in done
    ]
    universes: dict[tuple[int, int], tuple] = {}
    added = 0
    total = rebalancing.trials * len(sweep.sizes) * len(arms)
    progress = tqdm(
        work,
        desc="universe sizes",
        unit="run",
        total=total,
        initial=total - len(work),
        mininterval=PROGRESS_INTERVAL_TERMINAL
        if sys.stderr.isatty()
        else PROGRESS_INTERVAL_LOG,
    )
    for index, size, arm in progress:
        if (index, size) not in universes:
            trial, window = draw_rebalancing_trial(prices, index, config)
            sized = sized_trial(trial, asset_order(trial, config.seed), size)
            universe_path = write_universe(
                prices, sized, sweep.universe_dir / f"size_{size:04d}"
            )
            params = calibrated_cvar_params(
                prices[list(sized.assets)], window, rebalancing
            )
            universes = {(index, size): (sized, window, universe_path, params)}
        row = verified_backtest(arm, *universes[(index, size)], config, rates, services)
        new = pd.DataFrame([{**row, "size": size}])
        frame = new if frame is None else pd.concat([frame, new], ignore_index=True)
        write_results(sweep.output, frame, metadata)
        added += 1
        progress.set_postfix_str(
            f"trial {index} size {size} {arm.name} total "
            f"{row['total_return']:+.3f} {row['seconds']:.0f}s",
            refresh=False,
        )
    print(
        f"Universe sizes finished: {added} new runs, "
        f"{0 if frame is None else len(frame)} in {sweep.output}.",
        flush=True,
    )
    return frame


def complete_universe_runs(
    frame: pd.DataFrame, arms: Sequence[str], sizes: Sequence[int]
) -> pd.DataFrame:
    """Rows of the trials in which every arm finished at every size.

    Every size then reads the same trials, so a curve over the sizes changes
    only with the assets traded.

    Raises
    ------
    ValueError
        If fewer than ``MIN_TRIALS`` trials are complete.
    """
    frame = frame[frame["arm"].isin(arms) & frame["size"].isin(sizes)]
    counts = frame.groupby("trial").size()
    complete = counts.index[counts == len(arms) * len(sizes)]
    if len(complete) < MIN_TRIALS:
        raise ValueError(
            f"{len(complete)} trials have every arm's run at every universe size; "
            f"the blog needs at least {MIN_TRIALS}."
        )
    return frame[frame["trial"].isin(complete)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.simulator_comparison.universe_sizes",
        description="Rebalancing backtest of nested asset subsets of every "
        "trial's universe per simulator.",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="YAML configuration file."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    # Before cvxpy, cuOpt, or cuML is imported, so they see the GPU .env selects.
    apply_gpu_selection()
    run_universe_size_experiment(
        config, command=sys.argv if argv is None else ["main", *argv]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remove every output of a configuration, so its next run starts fresh.

The runners resume their results files, so a run after a change of code or
library defaults would refuse the old file or add to it. This deletes the
scenario results and report, and the rebalancing results, report, and
per-trial universes the configuration names, and prints each path it removes.
Run it as ``python -m experiments.simulator_comparison.clean``.
"""

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import ExperimentConfig, load_config


def outputs(config: ExperimentConfig) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """The results files and the directories a configuration writes."""
    rebalancing = config.rebalancing
    return (
        (config.output, rebalancing.output),
        (config.report_dir, rebalancing.report_dir, rebalancing.universe_dir),
    )


def remove_outputs(config: ExperimentConfig) -> list[Path]:
    """Delete the configuration's outputs that exist and return them.

    Raises
    ------
    ValueError
        If a results path is a directory or an output directory is a file,
        so a mistyped path never removes something the run did not write.
    """
    files, directories = outputs(config)
    wrong = [path for path in files if path.is_dir()]
    wrong += [path for path in directories if path.exists() and not path.is_dir()]
    if wrong:
        raise ValueError(f"Not removing {wrong}: not what the configuration writes.")
    removed = []
    for path in files:
        if path.exists():
            path.unlink()
            removed.append(path)
    for path in directories:
        if path.exists():
            shutil.rmtree(path)
            removed.append(path)
    return removed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.simulator_comparison.clean",
        description="Delete the results, reports, and universes a configuration "
        "writes, so its next run starts fresh.",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="YAML configuration file."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    removed = remove_outputs(load_config(args.config))
    for path in removed:
        print(f"Removed {path}")
    print(f"Clean finished: {len(removed)} outputs removed for {args.config}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

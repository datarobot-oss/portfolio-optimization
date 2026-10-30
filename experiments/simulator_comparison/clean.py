# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remove every output of a configuration, so its next run starts fresh.

The runners resume their results files, so a run after a change of code or
library defaults would refuse the old file or add to it. For a simulator
comparison this deletes the scenario results, report, and blog figures, the
rebalancing results, report, and per-trial universes, and the universe-size
results and universes the configuration names; for a CPU-against-GPU
configuration its results, report, and per-trial universes. It prints each
path it removes.
Run it as ``python -m experiments.simulator_comparison.clean``.
"""

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import DevicesConfig, ExperimentConfig, load_config, load_devices_config


def device_outputs(
    devices: DevicesConfig,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """The results file and the directories a CPU-against-GPU configuration writes."""
    return (devices.output,), (devices.report_dir, devices.universe_dir)


def outputs(config: ExperimentConfig) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """The results files and the directories a configuration writes."""
    rebalancing = config.rebalancing
    sweep = config.universe_sizes
    return (
        (config.output, rebalancing.output, sweep.output),
        (
            config.report_dir,
            config.blog_dir,
            rebalancing.report_dir,
            rebalancing.universe_dir,
            sweep.universe_dir,
        ),
    )


def remove_outputs(
    written: tuple[tuple[Path, ...], tuple[Path, ...]],
) -> list[Path]:
    """Delete the results files and directories of ``written`` that exist and return them.

    ``written`` is what ``outputs`` or ``device_outputs`` returns.

    Raises
    ------
    ValueError
        If a results path is a directory or an output directory is a file,
        so a mistyped path never removes something the run did not write.
    """
    files, directories = written
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
        description="Delete the results, reports, blog figures, and universes a "
        "configuration writes, so its next run starts fresh.",
    )
    configs = parser.add_mutually_exclusive_group(required=True)
    configs.add_argument(
        "--config", type=Path, help="YAML configuration of a simulator comparison."
    )
    configs.add_argument(
        "--devices-config",
        type=Path,
        help="YAML configuration of a CPU-against-GPU experiment; only its own "
        "outputs are removed, not those of the comparison it reads.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config is not None:
        path, written = args.config, outputs(load_config(args.config))
    else:
        devices, _ = load_devices_config(args.devices_config)
        path, written = args.devices_config, device_outputs(devices)
    removed = remove_outputs(written)
    for removed_path in removed:
        print(f"Removed {removed_path}")
    print(f"Clean finished: {len(removed)} outputs removed for {path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

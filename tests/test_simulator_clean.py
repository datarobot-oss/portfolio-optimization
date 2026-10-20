# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the command that removes a simulator-comparison config's outputs."""

from pathlib import Path

import pytest

from experiments.simulator_comparison.clean import outputs, remove_outputs
from experiments.simulator_comparison.config import load_config

CONFIG_DIR = Path(__file__).parents[1] / "experiments/simulator_comparison/configs"


@pytest.fixture()
def config(tmp_path, monkeypatch):
    # The shipped paths are relative to the repository root.
    monkeypatch.chdir(tmp_path)
    return load_config(CONFIG_DIR / "pilot.yaml")


def test_every_output_is_removed(config):
    files, directories = outputs(config)
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("rows")
    for path in directories:
        (path / "nested").mkdir(parents=True)
    removed = remove_outputs(config)
    assert removed == [*files, *directories]
    assert not any(path.exists() for path in removed)
    assert files[0].parent.exists()


def test_missing_outputs_are_no_error(config):
    assert remove_outputs(config) == []


def test_unexpected_path_types_are_left_alone(config):
    files, directories = outputs(config)
    files[0].mkdir(parents=True)
    directories[0].parent.mkdir(parents=True, exist_ok=True)
    directories[0].write_text("not a report")
    with pytest.raises(ValueError, match="Not removing"):
        remove_outputs(config)
    assert files[0].is_dir()
    assert directories[0].is_file()

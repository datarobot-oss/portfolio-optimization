# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAML configuration of the simulator comparison in experiments/."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from experiments.simulator_comparison.config import ExperimentConfig, load_config

CONFIG_DIR = Path(__file__).parents[1] / "experiments/simulator_comparison/configs"


def _shipped(name):
    return yaml.safe_load((CONFIG_DIR / name).read_text())


@pytest.mark.parametrize("name", ["pilot.yaml", "full.yaml", "timing.yaml"])
def test_shipped_configs_are_valid(name):
    config = load_config(CONFIG_DIR / name)
    assert "jointfm" in config.arms
    assert config.report_dir.name == f"{config.output.stem}_report"


def test_every_field_is_required():
    values = _shipped("pilot.yaml")
    del values["seed"]
    with pytest.raises(ValidationError, match="seed"):
        ExperimentConfig.model_validate(values)


def test_unknown_key_is_refused():
    values = _shipped("pilot.yaml") | {"jointfm_cpu_env": "jointfm-cpu.env"}
    with pytest.raises(ValidationError, match="jointfm_cpu_env"):
        ExperimentConfig.model_validate(values)


@pytest.mark.parametrize(
    "arms, message",
    [
        # The service .env selects decides the JointFM device, not the config.
        (["jointfm-gpu"], "Unknown arms"),
        (["kde-tpu"], "Unknown arms"),
        (["kde-cpu", "kde-cpu"], "only once"),
    ],
)
def test_arms_must_be_known_and_unique(arms, message):
    values = _shipped("pilot.yaml") | {"arms": arms}
    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(values)

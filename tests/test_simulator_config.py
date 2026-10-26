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


# Each set runs its base config and three twins (task experiments:pilot, :full).
SETS = ["pilot", "full"]
TWINS = ["next_day", "long_history", "risk_averse"]


@pytest.mark.parametrize(
    "name",
    [f"{base}.yaml" for base in SETS]
    + [f"{base}_{twin}.yaml" for base in SETS for twin in TWINS]
    + ["timing.yaml"],
)
def test_shipped_configs_are_valid(name):
    config = load_config(CONFIG_DIR / name)
    assert "jointfm" in config.arms
    assert config.report_dir.name == f"{config.output.stem}_report"


def _without_outputs(values):
    del values["output"], values["rebalancing"]["output"]
    return values


@pytest.mark.parametrize("base_name", SETS)
def test_long_history_twin_differs_only_in_look_back_and_outputs(base_name):
    base = _shipped(f"{base_name}.yaml")
    twin = _shipped(f"{base_name}_long_history.yaml")
    assert twin["look_back"] > base["look_back"]
    assert twin["look_back"] <= base["origin_history"]
    for values in (base, twin):
        del values["look_back"]
    assert _without_outputs(twin) == _without_outputs(base)


@pytest.mark.parametrize("base_name", SETS)
def test_next_day_twin_differs_only_in_horizons_and_outputs(base_name):
    """The twin's next day is a next-day-only request, read against its base's."""
    base = _shipped(f"{base_name}.yaml")
    twin = _shipped(f"{base_name}_next_day.yaml")
    assert twin["horizons"] == [1]
    assert base["horizons"][0] == 1 and len(base["horizons"]) > 1
    for values in (base, twin):
        del values["horizons"]
    assert _without_outputs(twin) == _without_outputs(base)


@pytest.mark.parametrize("base_name", SETS)
def test_risk_averse_twin_differs_only_in_risk_aversion_and_outputs(base_name):
    base = _shipped(f"{base_name}.yaml")
    twin = _shipped(f"{base_name}_risk_averse.yaml")
    base_cvar = base["rebalancing"]["cvar"]
    twin_cvar = twin["rebalancing"]["cvar"]
    assert twin_cvar["risk_aversion"] > base_cvar["risk_aversion"]
    for cvar in (base_cvar, twin_cvar):
        del cvar["risk_aversion"]
    assert _without_outputs(twin) == _without_outputs(base)


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


@pytest.mark.parametrize(
    "sizes, message",
    [
        ([1, 101], "exceed the 100 assets"),
        ([5, 2], "strictly increasing"),
        ([2, 2], "strictly increasing"),
        ([0, 2], "positive"),
    ],
)
def test_portfolio_sizes_must_increase_and_fit_the_universe(sizes, message):
    values = _shipped("pilot.yaml") | {"portfolio_sizes": sizes}
    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(values)


@pytest.mark.parametrize(
    "horizons, message",
    [
        ([5, 21], "start at 1"),
        ([1, 21, 5], "strictly increasing"),
        ([1, 1], "strictly increasing"),
        ([1, 64], "exceed the 63 forward returns"),
    ],
)
def test_horizons_start_at_the_next_day_and_fit_the_forward_window(horizons, message):
    """Horizons start at 1, increase, and end within the forward returns."""
    values = _shipped("pilot.yaml") | {"horizons": horizons}
    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(values)

# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arms: one scenario simulator on one device, and how they are resolved.

Every arm runs with the library's defaults, so the JointFM arm reads ``.env``
exactly as ``fit_type="jointfm"`` does for any user. Its device is where the
service ``.env`` selects computes, read from the service's health response:
a configuration names the arm ``jointfm``, and a run resolves it to
``jointfm-cpu`` or ``jointfm-gpu``. Comparing JointFM on both devices takes
two runs of one configuration, with ``.env`` pointing at the other service.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from portfolio_optimization import cvar_utils
from portfolio_optimization.settings import (
    JointFMSettings,
    KDESettings,
    ScenarioGenerationSettings,
)

Device = Literal["CPU", "GPU"]

# torch device type reported in a JointFM service's health ``device``
# (``cpu``, ``cuda``, ``cuda:1``) mapped to the arm device it serves.
SERVICE_DEVICE_TYPES = {"cpu": "CPU", "cuda": "GPU"}
# Configuration name of the JointFM arm, whose device the service decides.
JOINTFM_ARM = "jointfm"
# Service-record entries that must stay the same for every call of a session.
SERVICE_IDENTITY_KEYS = ("device", "model_version", "checkpoint_version")


@dataclass(frozen=True)
class Arm:
    """One simulator (a ``ScenarioGenerationSettings.fit_type``) on one device."""

    name: str
    fit_type: Literal["no_fit", "gaussian", "kde", "jointfm"]
    device: Device


ARMS = (
    Arm("no_fit-cpu", "no_fit", "CPU"),
    Arm("gaussian-cpu", "gaussian", "CPU"),
    Arm("kde-cpu", "kde", "CPU"),
    Arm("kde-gpu", "kde", "GPU"),
    Arm("jointfm-cpu", "jointfm", "CPU"),
    Arm("jointfm-gpu", "jointfm", "GPU"),
)
ARMS_BY_NAME = {arm.name: arm for arm in ARMS}
CONFIG_ARM_NAMES = (
    *(arm.name for arm in ARMS if arm.fit_type != "jointfm"),
    JOINTFM_ARM,
)


def scenario_settings(
    arm: Arm, *, num_scen: int, seed: int | None
) -> ScenarioGenerationSettings:
    """Settings that make ``generate_cvar_data`` run ``arm``.

    Every arm keeps the library's defaults apart from the KDE device, so the
    JointFM arm's client reads ``.env`` and ``config.yaml`` as for any user.
    """
    kde_settings = None
    jointfm_settings = None
    if arm.fit_type == "kde":
        kde_settings = KDESettings(device=arm.device)
    if arm.fit_type == "jointfm":
        jointfm_settings = JointFMSettings()
    return ScenarioGenerationSettings(
        num_scen=num_scen,
        fit_type=arm.fit_type,
        kde_settings=kde_settings,
        jointfm_settings=jointfm_settings,
        seed=seed,
    )


def describe_arms(arms: Sequence[Arm], *, num_scen: int) -> dict[str, dict]:
    """Each arm's settings apart from its seed and the client's file locations.

    The file locations are left out because the service they select is
    recorded by ``check_jointfm_service`` instead.
    """
    exclude = {
        "seed": True,
        "jointfm_settings": {"dotenv_path", "config_path"},
    }
    return {
        arm.name: scenario_settings(arm, num_scen=num_scen, seed=None).model_dump(
            mode="json", exclude=exclude
        )
        for arm in arms
    }


def check_jointfm_service() -> tuple[Arm, dict]:
    """The JointFM arm of the service the library's client settings select.

    The client is built from ``JointFMSettings()`` defaults, as the arm's
    calls build it, and asked for the service's health.

    Returns
    -------
    tuple
        The arm (``jointfm-cpu`` or ``jointfm-gpu``) and its service record:
        client version and the service's device, model, checkpoint,
        generation, image, and schema version.

    Raises
    ------
    ValueError
        If the service reports a device type other than CPU or CUDA.
    """
    settings = JointFMSettings()
    jointfm_client = cvar_utils._import_jointfm_client()
    health = jointfm_client.JointFMClient.from_env(
        dotenv_path=settings.dotenv_path, config_path=settings.config_path
    ).health()
    device_type = health.device.split(":")[0]
    if device_type not in SERVICE_DEVICE_TYPES:
        raise ValueError(
            f"The JointFM service reports device {health.device!r}; the comparison "
            f"knows {sorted(SERVICE_DEVICE_TYPES)}."
        )
    device = SERVICE_DEVICE_TYPES[device_type]
    return ARMS_BY_NAME[f"jointfm-{device.lower()}"], {
        "client_version": jointfm_client.__version__,
        "device": health.device,
        "model_version": health.model_version,
        "checkpoint_version": health.checkpoint_version,
        "generation": health.generation,
        "image_version": health.image_version,
        "schema_version": health.schema_version,
    }


def resolve_arms(names: Sequence[str]) -> tuple[tuple[Arm, ...], dict[str, dict]]:
    """The arms of configured names, with ``jointfm`` resolved by its service.

    Returns
    -------
    tuple
        The arms in configured order, and the JointFM service record by arm
        name (empty without a JointFM arm).
    """
    arms = []
    services = {}
    for name in names:
        if name == JOINTFM_ARM:
            arm, service = check_jointfm_service()
            services[arm.name] = service
        else:
            arm = ARMS_BY_NAME[name]
        arms.append(arm)
    return tuple(arms), services


def verify_jointfm_service(arm: Arm, service: dict) -> None:
    """Refuse a JointFM call when ``.env`` now selects another service.

    The client reads ``.env`` again on every call, so a selector changed
    while a session runs would send the arm's remaining calls to a service on
    another device, or with another checkpoint, without any error.

    Raises
    ------
    ValueError
        If the selected service's arm, device, model, or checkpoint differs
        from the one recorded for ``arm`` at session start.
    """
    current_arm, current = check_jointfm_service()
    changed = {
        key: (service[key], current[key])
        for key in SERVICE_IDENTITY_KEYS
        if service[key] != current[key]
    }
    if current_arm != arm or changed:
        raise ValueError(
            f"The JointFM service .env selects changed during the {arm.name} "
            f"session (recorded, now): {changed}. Rerun the session with one "
            f"selector."
        )


def check_same_checkpoint(services: dict[str, dict]) -> None:
    """Refuse JointFM arms whose services serve different checkpoints.

    Raises
    ------
    ValueError
        If two services report different model or checkpoint versions.
    """
    checkpoints = {
        (service["model_version"], service["checkpoint_version"])
        for service in services.values()
    }
    if len(checkpoints) > 1:
        raise ValueError(
            f"The JointFM arms' services serve different checkpoints: {services}."
        )

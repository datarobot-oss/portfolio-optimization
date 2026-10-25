# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arms: one scenario simulator on one device, and how they are resolved.

Every arm runs with the library's defaults, so the JointFM arm reads ``.env``
exactly as ``fit_type="jointfm"`` does for any user. Its device is where the
service ``.env`` selects computes, read from the service's health response:
a configuration names the arm ``jointfm``, and a run resolves it to
``jointfm-cpu`` or ``jointfm-gpu``. Comparing JointFM on both devices takes
two runs of one configuration, with ``.env`` pointing at the other service.

The dependence swap cells (``SwapArm``, built in ``swaps.py``), the
historical-mean JointFM arm (``HistoricalMeanArm``), and the independence
control (``IndependentArm``, built in ``independence.py``) are *derived arms*:
they reuse configured simulators' scenarios, their *donors*, instead of
simulating, and run only next to them.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

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
# Configuration name of the arm whose scenarios are the look-back window's rows.
WINDOW_ARM = "no_fit-cpu"
# Each dependence swap cell (swaps.py) by name: its marginal donor and its
# dependence donor, as configured arm names.
SWAP_DONORS = {
    "swap-jointfm-window": (JOINTFM_ARM, WINDOW_ARM),
    "swap-window-jointfm": (WINDOW_ARM, JOINTFM_ARM),
    "swap-window-window": (WINDOW_ARM, WINDOW_ARM),
}
# Configuration name of the arm with JointFM's scenarios and the window's mean.
HISTORICAL_MEAN_ARM = "jointfm-historical"
# Configuration name of the positive control: the window's rows permuted
# independently per asset.
INDEPENDENT_ARM = "window-independent"
# Every derived arm by name: the configured arms it draws from.
DERIVED_DONORS = {
    **SWAP_DONORS,
    HISTORICAL_MEAN_ARM: (JOINTFM_ARM,),
    INDEPENDENT_ARM: (WINDOW_ARM,),
}
CONFIG_ARM_NAMES = (
    *(arm.name for arm in ARMS if arm.fit_type != "jointfm"),
    JOINTFM_ARM,
    *DERIVED_DONORS,
)


@dataclass(frozen=True)
class SwapArm:
    """A dependence swap cell: its marginal donor's values in its dependence donor's ranks.

    Results rows and figures tell the cells apart by name, as they tell the
    simulators apart by fit type, so a cell's fit type is its name. The
    shuffle runs on the CPU, whatever device its donors computed on.
    """

    name: str
    marginals: Arm
    dependence: Arm
    device: ClassVar[Device] = "CPU"

    @property
    def fit_type(self) -> str:
        return self.name

    @property
    def donors(self) -> tuple[Arm, ...]:
        """The distinct simulators the cell draws from, marginal donor first."""
        return tuple(dict.fromkeys((self.marginals, self.dependence)))


@dataclass(frozen=True)
class HistoricalMeanArm:
    """JointFM's scenarios with the look-back window's mean as expected returns.

    It runs ``JointFMSettings(expected_return="historical")`` with its donor's
    seed, so it draws exactly its donor's scenarios and differs from it only
    in ``CvarData.mean``, which is the value every other simulator uses. Its
    device is its donor's, where its scenarios are computed.

    Raises
    ------
    ValueError
        If the donor is not a JointFM arm, the only simulator whose expected
        returns are not already the window's mean.
    """

    name: str
    scenarios: Arm

    def __post_init__(self):
        if self.scenarios.fit_type != "jointfm":
            raise ValueError(
                f"{self.name} takes a JointFM arm's scenarios; got {self.scenarios.name}."
            )

    @property
    def fit_type(self) -> str:
        return self.name

    @property
    def device(self) -> Device:
        return self.scenarios.device

    @property
    def donors(self) -> tuple[Arm, ...]:
        return (self.scenarios,)


@dataclass(frozen=True)
class IndependentArm:
    """The positive control: the window's scenarios permuted independently per asset.

    It keeps every asset's scenario values, and so its distribution and its
    mean, and removes all dependence between assets; the permutation runs on
    the CPU (``independence.py``).
    """

    name: str
    window: Arm
    device: ClassVar[Device] = "CPU"

    @property
    def fit_type(self) -> str:
        return self.name

    @property
    def donors(self) -> tuple[Arm, ...]:
        return (self.window,)


AnyArm = Arm | SwapArm | HistoricalMeanArm | IndependentArm


def simulators(arm: AnyArm) -> tuple[Arm, ...]:
    """The simulators ``arm``'s scenarios come from: itself, or a derived arm's donors."""
    return (arm,) if isinstance(arm, Arm) else arm.donors


def jointfm_calls(arm: AnyArm) -> tuple[Arm, ...]:
    """The JointFM arms whose service ``arm``'s scenarios call."""
    return tuple(a for a in simulators(arm) if a.fit_type == "jointfm")


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


def historical_mean_settings(
    arm: HistoricalMeanArm, *, num_scen: int, seed: int | None
) -> ScenarioGenerationSettings:
    """Settings that make ``generate_cvar_data`` run ``arm``: its donor's, on the window's mean."""
    settings = scenario_settings(arm.scenarios, num_scen=num_scen, seed=seed)
    settings.jointfm_settings = settings.jointfm_settings.model_copy(
        update={"expected_return": "historical"}
    )
    return settings


def describe_arms(arms: Sequence[AnyArm], *, num_scen: int) -> dict[str, dict]:
    """Each arm's settings apart from its seed and the client's file locations.

    The file locations are left out because the service they select is
    recorded by ``check_jointfm_service`` instead. A derived arm is described
    by its resolved donors, so a session whose JointFM arm runs on the other
    device cannot add rows to a derived arm recorded with the first.
    """
    exclude = {
        "seed": True,
        "jointfm_settings": {"dotenv_path", "config_path"},
    }

    def describe(arm: AnyArm) -> dict:
        if isinstance(arm, SwapArm):
            return {
                "marginals": arm.marginals.name,
                "dependence": arm.dependence.name,
                "num_scen": num_scen,
            }
        if isinstance(arm, HistoricalMeanArm):
            return {
                "scenarios": arm.scenarios.name,
                **historical_mean_settings(
                    arm, num_scen=num_scen, seed=None
                ).model_dump(mode="json", exclude=exclude),
            }
        if isinstance(arm, IndependentArm):
            return {"window": arm.window.name}
        return scenario_settings(arm, num_scen=num_scen, seed=None).model_dump(
            mode="json", exclude=exclude
        )

    return {arm.name: describe(arm) for arm in arms}


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


def check_donors(names: Sequence[str]) -> None:
    """Refuse a derived arm whose donors are not configured next to it.

    Raises
    ------
    ValueError
        If a configured derived arm's donor is not among ``names``.
    """
    missing = {
        name: [donor for donor in DERIVED_DONORS[name] if donor not in names]
        for name in names
        if name in DERIVED_DONORS
    }
    missing = {name: donors for name, donors in missing.items() if donors}
    if missing:
        raise ValueError(
            f"Derived arms run next to their donors; configure {missing} as well."
        )


def resolve_arms(
    names: Sequence[str],
) -> tuple[tuple[AnyArm, ...], dict[str, dict]]:
    """The arms of configured names, with ``jointfm`` resolved by its service.

    A derived arm's donors resolve to the configured simulators of their names.

    Returns
    -------
    tuple
        The arms in configured order, and the JointFM service record by arm
        name (empty without a JointFM arm).

    Raises
    ------
    ValueError
        As ``check_donors`` and ``check_jointfm_service`` raise.
    """
    check_donors(names)
    simulators = {}
    services = {}
    for name in names:
        if name == JOINTFM_ARM:
            arm, service = check_jointfm_service()
            services[arm.name] = service
            simulators[name] = arm
        elif name not in DERIVED_DONORS:
            simulators[name] = ARMS_BY_NAME[name]

    def resolve(name: str) -> AnyArm:
        if name in SWAP_DONORS:
            marginals, dependence = SWAP_DONORS[name]
            return SwapArm(name, simulators[marginals], simulators[dependence])
        if name == HISTORICAL_MEAN_ARM:
            return HistoricalMeanArm(name, simulators[JOINTFM_ARM])
        if name == INDEPENDENT_ARM:
            return IndependentArm(name, simulators[WINDOW_ARM])
        return simulators[name]

    return tuple(resolve(name) for name in names), services


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

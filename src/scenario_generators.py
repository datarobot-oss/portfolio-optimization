# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scenario generators that ``ScenarioGenerationSettings.fit_type`` names.

A scenario generator turns a look-back window into return scenarios for CVaR
optimization. The library ships ``gaussian``, ``kde`` and ``no_fit``; any other
package adds one under its own name, either by calling :func:`register` or,
so that ``fit_type="<name>"`` works as soon as the package is installed, by
declaring an entry point::

    [project.entry-points."portfolio_optimization.scenario_generators"]
    my_model = "my_package:generate"

A generator is called as ``generator(returns_dict, settings)``: ``returns_dict``
is what ``utils.calculate_returns`` builds (``returns``, ``mean``,
``covariance``, ``return_type``, ``freq``, ...) and ``settings`` is the
``ScenarioGenerationSettings``, whose ``generator_options`` carry the
generator's own options. It returns :class:`Scenarios`. An optional
``display_name`` attribute names it in plot titles.
"""

from importlib.metadata import entry_points
from typing import TYPE_CHECKING, NamedTuple, Optional, Protocol

import numpy as np

if TYPE_CHECKING:
    from .settings import ScenarioGenerationSettings

ENTRY_POINT_GROUP = "portfolio_optimization.scenario_generators"


class Scenarios(NamedTuple):
    """What a scenario generator returns.

    Attributes
    ----------
    R : np.ndarray
        Scenario returns, shape (num_scenarios, n_assets), columns in the order
        of ``returns_dict["returns"]``.
    mean : np.ndarray, optional
        Expected return per asset for ``CvarData.mean``. None uses the
        look-back window's mean, ``returns_dict["mean"]``.
    p : np.ndarray, optional
        Scenario probabilities, shape (num_scenarios,). None weights the
        scenarios equally.
    """

    R: np.ndarray
    mean: Optional[np.ndarray] = None
    p: Optional[np.ndarray] = None


class ScenarioGenerator(Protocol):
    def __call__(
        self, returns_dict: dict, settings: "ScenarioGenerationSettings"
    ) -> Scenarios: ...


def _gaussian(returns_dict, settings):
    # default_rng(None) draws fresh entropy, matching the previous global-RNG
    # behaviour when no seed is configured.
    rng = np.random.default_rng(settings.seed)
    return Scenarios(
        rng.multivariate_normal(
            returns_dict["mean"], returns_dict["covariance"], size=settings.num_scen
        )
    )


def _kde(returns_dict, settings):
    from .cvar_utils import generate_samples_kde  # cvar_utils imports this module

    return Scenarios(
        generate_samples_kde(
            settings.num_scen,
            returns_dict["returns"],
            kde_settings=settings.kde_settings,
            verbose=settings.verbose,
            seed=settings.seed,
        )
    )


def _no_fit(returns_dict, settings):
    # The window's rows are the scenarios, so num_scen is the window's length.
    return Scenarios(np.asarray(returns_dict["returns"]))


_gaussian.display_name = "Gaussian"
_kde.display_name = "KDE"
_no_fit.display_name = "Historical"

_REGISTRY: dict[str, ScenarioGenerator] = {
    "gaussian": _gaussian,
    "kde": _kde,
    "no_fit": _no_fit,
}


def register(name: str, generator: ScenarioGenerator) -> None:
    """Make ``generator`` available as ``fit_type=name``, replacing any other."""
    _REGISTRY[name] = generator


def available() -> list[str]:
    """Names of the built-in, registered and installed scenario generators."""
    return sorted({*_REGISTRY, *entry_points(group=ENTRY_POINT_GROUP).names})


def get(name: str) -> ScenarioGenerator:
    """The scenario generator ``fit_type=name`` selects.

    Raises
    ------
    ValueError
        If no built-in, registered or installed generator has that name.
    """
    if name not in _REGISTRY:
        for entry_point in entry_points(group=ENTRY_POINT_GROUP, name=name):
            _REGISTRY[name] = entry_point.load()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown fit_type {name!r}; available scenario generators: "
            f"{available()}. A plugin's generator is available once its "
            f"package is installed."
        )
    return _REGISTRY[name]


def display_name(name: str) -> str:
    """How plot titles name the generator ``fit_type=name`` selects."""
    return getattr(get(name), "display_name", name)

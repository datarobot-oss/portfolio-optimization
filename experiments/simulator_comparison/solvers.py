# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The solvers an optimization runs on, by configuration name.

Each solver is a CVXPY backend with fixed options. Its device is where its
algorithm computes: cuOpt's PDLP and barrier methods run on the GPU and its
dual simplex on the CPU (cuOpt user guide, LP features), while HiGHS and
Clarabel run on the CPU. Every cuOpt method needs a visible GPU to load,
whatever device its algorithm computes on, so a process that hides every GPU
can run only the solvers without ``needs_gpu``.

Backends and options are plain strings, so naming a solver imports nothing:
importing cvxpy creates a CUDA context, which must wait until the process's GPU
selection is applied (``portfolio_optimization.gpu.apply_gpu_selection``).
"""

from dataclasses import dataclass

from .arms import Device


@dataclass(frozen=True)
class Solver:
    """A CVXPY backend with fixed options, and where its algorithm computes.

    Attributes
    ----------
    name : str
        Configuration name.
    label : str
        Name in figures.
    backend : str
        CVXPY solver name, passed as ``solver`` to ``cvxpy.Problem.solve``.
    device : {"CPU", "GPU"}
        Where the algorithm computes.
    needs_gpu : bool
        Whether the backend loads only where a GPU is visible.
    options : tuple of (str, str)
        Further keyword arguments of ``cvxpy.Problem.solve``.
    module : str
        Import name of the distribution the backend runs in, whose version a
        results file records.
    """

    name: str
    label: str
    backend: str
    device: Device
    needs_gpu: bool
    options: tuple[tuple[str, str], ...]
    module: str

    def settings(self) -> dict:
        """Keyword arguments of ``cvxpy.Problem.solve`` that run this solver."""
        return {"solver": self.backend, "verbose": False, **dict(self.options)}


# cuOpt's method option names a member of cuopt's SolverMethod and its other
# options name cuOpt parameters; CVXPY ignores a name it cannot resolve, so
# check_cuopt_options verifies them before a run.
SOLVERS = (
    Solver(
        "CUOPT",
        "cuOpt PDLP",
        "CUOPT",
        "GPU",
        True,
        (("solver_method", "PDLP"),),
        "cuopt",
    ),
    Solver(
        "CUOPT_BARRIER",
        "cuOpt barrier",
        "CUOPT",
        "GPU",
        True,
        # Every scenario row holds every weight, so the default normal
        # equations are dense in the scenarios and exhaust GPU memory.
        (("solver_method", "Barrier"), ("augmented", "1")),
        "cuopt",
    ),
    Solver(
        "CUOPT_DUAL_SIMPLEX",
        "cuOpt dual simplex",
        "CUOPT",
        "CPU",
        True,
        (("solver_method", "DualSimplex"),),
        "cuopt",
    ),
    Solver("HIGHS", "HiGHS", "HIGHS", "CPU", False, (), "highspy"),
    Solver("CLARABEL", "Clarabel", "CLARABEL", "CPU", False, (), "clarabel"),
)
SOLVERS_BY_NAME = {solver.name: solver for solver in SOLVERS}
SOLVER_NAMES = tuple(SOLVERS_BY_NAME)
CUOPT_BACKEND = "CUOPT"


def solver_settings(name: str) -> dict:
    """Keyword arguments of ``cvxpy.Problem.solve`` that run the solver ``name``."""
    return SOLVERS_BY_NAME[name].settings()


def check_cuopt_options(names: tuple[str, ...]) -> None:
    """Refuse a cuOpt solver whose method or option name cuOpt does not know.

    CVXPY drops a method name it cannot resolve and every option that is not
    a cuOpt parameter, then solves with cuOpt's defaults instead, so a
    misspelled name would time another configuration under this solver's
    name. Imports cuOpt, so call it only after the process's GPU selection is
    applied.

    Raises
    ------
    ValueError
        If a named cuOpt solver's ``solver_method`` is not a member of
        cuOpt's ``SolverMethod``, or another of its options is not a cuOpt
        parameter.
    """
    options = {
        name: dict(SOLVERS_BY_NAME[name].options)
        for name in names
        if SOLVERS_BY_NAME[name].backend == CUOPT_BACKEND
    }
    if not options:
        return
    from cuopt.linear_programming.solver import solver_parameters
    from cuopt.linear_programming.solver_settings import SolverMethod

    unknown_methods = {
        name: chosen["solver_method"]
        for name, chosen in options.items()
        if chosen["solver_method"] not in SolverMethod.__members__
    }
    if unknown_methods:
        raise ValueError(
            f"cuOpt knows no method {unknown_methods}; choose from "
            f"{list(SolverMethod.__members__)}."
        )
    parameters = {
        value
        for key, value in vars(solver_parameters).items()
        if key.startswith("CUOPT_") and isinstance(value, str)
    }
    unknown_options = {
        name: sorted(set(chosen) - {"solver_method"} - parameters)
        for name, chosen in options.items()
        if set(chosen) - {"solver_method"} - parameters
    }
    if unknown_options:
        raise ValueError(f"cuOpt knows no option {unknown_options}.")

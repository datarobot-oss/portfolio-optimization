# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Solver registry of the CPU-against-GPU experiment and its cuOpt option check."""

import dataclasses

import pytest

from experiments.simulator_comparison import solvers


def test_cuopt_barrier_solves_the_augmented_system():
    """cuOpt barrier factors the augmented system, not the normal equations.

    Every scenario row of the CVaR problem holds every portfolio weight, so
    the normal equations are dense in the scenarios: at 50,000 scenarios
    cuOpt's default asked for 1.2 TB of GPU memory and reported a numerical
    error.
    """
    assert solvers.solver_settings("CUOPT_BARRIER")["augmented"] == "1"


def test_cuopt_option_check_accepts_every_registered_solver():
    """Every registered cuOpt method and option name is one cuOpt knows."""
    pytest.importorskip("cuopt", reason="cuOpt runtime required")
    solvers.check_cuopt_options(solvers.SOLVER_NAMES)


@pytest.mark.parametrize(
    ("option", "message"),
    [
        (("solver_method", "Barier"), "knows no method"),
        (("augmentd", "1"), "knows no option"),
    ],
)
def test_cuopt_option_check_refuses_unknown_names(monkeypatch, option, message):
    """A misspelled method or option, which CVXPY would drop silently, is refused."""
    pytest.importorskip("cuopt", reason="cuOpt runtime required")
    barrier = solvers.SOLVERS_BY_NAME["CUOPT_BARRIER"]
    options = dict(barrier.options) | dict([option])
    monkeypatch.setitem(
        solvers.SOLVERS_BY_NAME,
        "CUOPT_BARRIER",
        dataclasses.replace(barrier, options=tuple(options.items())),
    )
    with pytest.raises(ValueError, match=message):
        solvers.check_cuopt_options(("CUOPT_BARRIER",))

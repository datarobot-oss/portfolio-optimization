# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Random portfolios of the simulator comparison in experiments/: which cells
the bounds allow, and that every draw keeps the bounds."""

import numpy as np
import pytest

from experiments.simulator_comparison.config import CvarConfig
from experiments.simulator_comparison.portfolios import (
    CAP_TOLERANCE,
    draw_portfolios,
    feasible_kinds,
    portfolio_cells,
    portfolio_returns,
)

# The bounds of the shipped configs.
BOUNDS = CvarConfig(
    w_min=-0.3,
    w_max=0.8,
    c_min=0.1,
    c_max=0.2,
    L_tar=1.6,
    T_tar=0.5,
    confidence=0.95,
    risk_aversion=1.0,
)
SIZES = (1, 2, 3, 10, 40)
SUBSETS = 50
N_ASSETS = 40


@pytest.fixture()
def weights():
    cells = portfolio_cells(SIZES, BOUNDS)
    drawn = draw_portfolios(np.random.default_rng(0), N_ASSETS, cells, SUBSETS, BOUNDS)
    return {
        cell: drawn[position * SUBSETS : (position + 1) * SUBSETS]
        for position, cell in enumerate(cells)
    }


class TestCells:
    def test_bounds_decide_the_kinds_per_size(self):
        # One long position of at most 0.8 cannot carry 0.8 of capital plus a
        # short leg, so long-short starts at two longs and one short.
        assert feasible_kinds(1, BOUNDS) == ("equal", "long_only")
        assert feasible_kinds(2, BOUNDS) == ("equal", "long_only")
        assert feasible_kinds(3, BOUNDS) == ("equal", "long_only", "long_short")

    def test_long_short_needs_shorts_allowed(self):
        long_only = BOUNDS.model_copy(update={"w_min": 0.0})
        assert feasible_kinds(40, long_only) == ("equal", "long_only")

    def test_cells_are_ordered_by_size_then_kind(self):
        assert portfolio_cells((2, 3), BOUNDS) == (
            ("equal", 2),
            ("long_only", 2),
            ("equal", 3),
            ("long_only", 3),
            ("long_short", 3),
        )


class TestDraws:
    def test_every_portfolio_holds_its_size(self, weights):
        for (_, size), rows in weights.items():
            assert ((rows != 0).sum(axis=1) == size).all()

    def test_equal_weights_are_fully_invested(self, weights):
        for size in SIZES:
            rows = weights[("equal", size)]
            np.testing.assert_allclose(rows.sum(axis=1), 1.0)
            assert set(np.round(rows[rows != 0], 12)) == {round(1 / size, 12)}

    @pytest.mark.parametrize("kind", ["long_only", "long_short"])
    def test_random_kinds_keep_every_bound(self, weights, kind):
        for (drawn_kind, _), rows in weights.items():
            if drawn_kind != kind:
                continue
            cash = 1 - rows.sum(axis=1)
            assert (rows >= BOUNDS.w_min - CAP_TOLERANCE).all()
            assert (rows <= BOUNDS.w_max + CAP_TOLERANCE).all()
            assert (cash >= BOUNDS.c_min - CAP_TOLERANCE).all()
            assert (cash <= BOUNDS.c_max + CAP_TOLERANCE).all()
            assert (np.abs(rows).sum(axis=1) <= BOUNDS.L_tar + CAP_TOLERANCE).all()

    def test_long_only_holds_no_short_and_long_short_at_least_one(self, weights):
        for (kind, _), rows in weights.items():
            if kind == "long_only":
                assert (rows >= 0).all()
            if kind == "long_short":
                assert (rows.min(axis=1) < 0).all()

    def test_single_asset_takes_the_largest_weight_the_bounds_allow(self, weights):
        rows = weights[("long_only", 1)]
        np.testing.assert_allclose(rows.sum(axis=1), BOUNDS.w_max)

    def test_equal_seeds_draw_equal_portfolios(self):
        cells = portfolio_cells(SIZES, BOUNDS)

        def draw(seed):
            return draw_portfolios(
                np.random.default_rng(seed), N_ASSETS, cells, 3, BOUNDS
            )

        np.testing.assert_array_equal(draw(1), draw(1))
        assert not np.array_equal(draw(1), draw(2))

    def test_size_above_the_universe_is_refused(self):
        with pytest.raises(ValueError, match="does not fit"):
            draw_portfolios(
                np.random.default_rng(0), 4, [("equal", 5)], SUBSETS, BOUNDS
            )


class TestReturns:
    def test_portfolio_return_is_the_weighted_sum_of_asset_returns(self):
        rng = np.random.default_rng(3)
        returns = rng.normal(0.0, 0.02, size=(100, 4))
        weights = np.array([[0.25, 0.25, 0.25, 0.25], [0.7, 0.0, 0.3, 0.0]])
        cash = 1 - weights.sum(axis=1, keepdims=True)
        # Each position's value moves with its asset; cash keeps its value.
        values = (weights[:, None, :] * (1 + returns)).sum(axis=2) + cash
        np.testing.assert_allclose(portfolio_returns(returns, weights), values - 1)

    def test_short_position_can_lose_more_than_the_capital(self):
        """Short 1.0 of an asset that triples while the long side is flat."""
        weights = np.array([[2.0, -1.0]])
        np.testing.assert_allclose(
            portfolio_returns(np.array([[0.0, 2.0]]), weights), [[-2.0]]
        )

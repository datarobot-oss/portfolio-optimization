# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Random portfolios that the portfolio scores evaluate, and their returns.

A *cell* is one portfolio kind at one portfolio size, the number of assets the
portfolio holds. At each size, every subset of the universe carries one
portfolio of each kind the bounds allow:

- ``equal``: equal weights, fully invested and without cash, at every size;
  it ignores the bounds, so it matches the equal-weight reading of
  ``ScenarioScores`` at the full universe.
- ``long_only``: random non-negative weights within the weight, cash, and
  leverage bounds of the rebalancing optimization (``CvarConfig``).
- ``long_short``: random weights within the same bounds with at least one
  short position.

A kind is left out at a size the bounds cannot hold, such as a long-short
portfolio whose long side cannot carry the invested capital, so the cells
follow from the sizes and the bounds alone and are recorded in the run
identity.
"""

import math
from collections.abc import Sequence

import numpy as np

from .config import CvarConfig

KINDS = ("equal", "long_only", "long_short")
# Separates the portfolio draws from the scenario draws of the same seed.
PORTFOLIO_STREAM = 0x504F5254
# Slack for float rounding in the weight caps.
CAP_TOLERANCE = 1e-12


def _long_only_cash_floor(size: int, bounds: CvarConfig) -> float:
    return max(bounds.c_min, 1 - size * bounds.w_max, 1 - bounds.L_tar)


def _fewest_longs(bounds: CvarConfig) -> int:
    """Fewest long positions that carry the invested capital with room for shorts."""
    return math.floor((1 - bounds.c_max) / bounds.w_max) + 1


def feasible_kinds(size: int, bounds: CvarConfig) -> tuple[str, ...]:
    """The kinds of ``KINDS`` that a portfolio of ``size`` assets can take.

    A long-only portfolio needs a cash weight in ``[c_min, c_max]`` whose
    complement fits ``size`` weights of at most ``w_max`` and the leverage
    bound. A long-short portfolio additionally needs ``w_min < 0``, at least
    one short position, and long positions whose caps exceed the invested
    capital, so that a short leg can be added below the leverage bound.
    """
    kinds = ["equal"]
    if _long_only_cash_floor(size, bounds) <= bounds.c_max:
        kinds.append("long_only")
    if (
        bounds.w_min < 0
        and bounds.L_tar > 1 - bounds.c_max
        and _fewest_longs(bounds) <= size - 1
    ):
        kinds.append("long_short")
    return tuple(kinds)


def portfolio_cells(
    sizes: Sequence[int], bounds: CvarConfig
) -> tuple[tuple[str, int], ...]:
    """Every (kind, size) cell, by size and then in ``KINDS`` order."""
    return tuple(
        (kind, size) for size in sizes for kind in feasible_kinds(size, bounds)
    )


def _capped_split(total: float, count: int, cap: float, rng) -> np.ndarray:
    """``count`` random non-negative weights that sum to ``total``, each at most ``cap``.

    Draws a uniform point of the simplex and moves every weight above ``cap``
    down to it, spreading the excess over the uncapped weights in proportion
    to their size. Each pass caps at least one more weight, so ``count``
    passes always end.

    Raises
    ------
    ValueError
        If ``count`` weights capped at ``cap`` cannot sum to ``total``.
    """
    if total > count * cap + CAP_TOLERANCE:
        raise ValueError(f"{count} weights of at most {cap} cannot sum to {total}.")
    weights = total * rng.dirichlet(np.ones(count))
    for _ in range(count):
        over = weights > cap
        if not over.any():
            break
        excess = (weights[over] - cap).sum()
        weights[over] = cap
        free = weights < cap
        weights[free] += excess * weights[free] / weights[free].sum()
    return np.minimum(weights, cap)


def _equal(size: int, bounds: CvarConfig, rng) -> np.ndarray:
    return np.full(size, 1.0 / size)


def _long_only(size: int, bounds: CvarConfig, rng) -> np.ndarray:
    cash = rng.uniform(_long_only_cash_floor(size, bounds), bounds.c_max)
    return _capped_split(1 - cash, size, bounds.w_max, rng)


def _long_short(size: int, bounds: CvarConfig, rng) -> np.ndarray:
    longs = int(rng.integers(_fewest_longs(bounds), size))
    shorts = size - longs
    cash = rng.uniform(
        max(bounds.c_min, 1 - longs * bounds.w_max, 1 - bounds.L_tar), bounds.c_max
    )
    net = 1 - cash
    gross = rng.uniform(
        net,
        min(
            bounds.L_tar,
            net - 2 * shorts * bounds.w_min,
            2 * longs * bounds.w_max - net,
        ),
    )
    long_leg = _capped_split((gross + net) / 2, longs, bounds.w_max, rng)
    short_leg = _capped_split((gross - net) / 2, shorts, -bounds.w_min, rng)
    weights = np.concatenate([long_leg, -short_leg])
    return weights[rng.permutation(size)]


_DRAWS = {"equal": _equal, "long_only": _long_only, "long_short": _long_short}


def draw_portfolios(
    rng: np.random.Generator,
    n_assets: int,
    cells: Sequence[tuple[str, int]],
    subsets: int,
    bounds: CvarConfig,
) -> np.ndarray:
    """Asset weights of every portfolio of ``cells``, ``subsets`` per cell.

    At each size, ``subsets`` random asset subsets are drawn, and each carries
    one portfolio per kind of ``cells`` at that size. Weights outside a
    portfolio's subset are zero, and its cash weight is one minus the sum of
    its asset weights.

    Parameters
    ----------
    rng : np.random.Generator
        Draws subsets and weights; equal seeds give equal portfolios.
    n_assets : int
        Assets in the universe.
    cells : sequence of (str, int)
        (kind, size) cells as ``portfolio_cells`` returns them.
    subsets : int
        Asset subsets per size.
    bounds : CvarConfig
        Weight, cash, and leverage bounds of the random kinds.

    Returns
    -------
    np.ndarray
        Weights, shape ``(len(cells) * subsets, n_assets)``: the rows of cell
        ``c`` are ``c * subsets`` to ``(c + 1) * subsets - 1``.

    Raises
    ------
    ValueError
        If a cell's size exceeds ``n_assets`` or its kind is unknown.
    """
    sizes = sorted({size for _, size in cells})
    if sizes[-1] > n_assets:
        raise ValueError(
            f"A portfolio of {sizes[-1]} assets does not fit a universe of {n_assets}."
        )
    members = {
        size: [rng.choice(n_assets, size, replace=False) for _ in range(subsets)]
        for size in sizes
    }
    weights = np.zeros((len(cells) * subsets, n_assets))
    for position, (kind, size) in enumerate(cells):
        if kind not in _DRAWS:
            raise ValueError(f"Unknown portfolio kind {kind!r}; known: {KINDS}.")
        for subset, assets in enumerate(members[size]):
            weights[position * subsets + subset, assets] = _DRAWS[kind](
                size, bounds, rng
            )
    return weights


def portfolio_log_returns(log_returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Log return of each portfolio for each row of asset log returns.

    A portfolio's simple return is its weighted sum of the assets' simple
    returns; cash earns nothing, which shifts a day's scenarios and its
    outcome alike.

    Parameters
    ----------
    log_returns : np.ndarray
        Asset log returns, shape ``(rows, n_assets)``.
    weights : np.ndarray
        Portfolio weights, shape ``(portfolios, n_assets)``.

    Returns
    -------
    np.ndarray
        Portfolio log returns, shape ``(portfolios, rows)``.

    Raises
    ------
    ValueError
        If a portfolio loses its whole value, where the log return is undefined.
    """
    simple = weights @ np.expm1(log_returns).T
    if (simple <= -1).any():
        raise ValueError("A portfolio return reaches -100 %; its log is undefined.")
    return np.log1p(simple)

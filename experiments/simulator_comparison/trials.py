# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trials: seeded draws of an asset universe and of forecast origins.

A trial's draw depends only on the experiment seed and the trial's index, so
trial ``i`` is the same in every run and on every host, and adding trials to a
run never changes the trials already in it.
"""

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_optimization.settings import ReturnsComputeSettings
from portfolio_optimization.utils import calculate_returns

# Simple returns aggregate across positions, so portfolio returns, the optimizer's
# CVaR, and the backtest's values hold for short positions; JointFM still
# forecasts log prices and converts (generate_horizon_samples_jointfm).
RETURNS_SETTINGS = ReturnsComputeSettings(return_type="LINEAR", freq=1)


@dataclass(frozen=True)
class Trial:
    """One draw of assets and forecast origins shared by every arm.

    Attributes
    ----------
    index : int
        Trial index; with the experiment seed it determines the whole draw.
    assets : tuple of str
        Asset columns of the price table, in the table's column order.
    origins : tuple of int
        Row positions in the price table of each forecast origin, ascending.
        The origin row is the last price of its look-back window, and the row
        after it is the realized outcome.
    """

    index: int
    assets: tuple[str, ...]
    origins: tuple[int, ...]

    @property
    def assets_hash(self) -> str:
        """Short digest of the ordered asset list, to group rows by universe."""
        return hashlib.sha256("\n".join(self.assets).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class OriginInputs:
    """What every arm receives at one forecast origin of one trial.

    Attributes
    ----------
    origin : pd.Timestamp
        Date of the last price in the look-back window.
    returns_dict : dict
        The window's returns as ``calculate_returns`` builds them for the
        optimizer; ``generate_cvar_data`` adds ``cvar_data`` to a copy.
    forward : pd.DataFrame
        Daily simple returns of the rows after the origin, oldest first. Only
        scoring reads them; no simulator sees them.
    seed : int
        Scenario seed passed to every arm's simulator.
    """

    origin: pd.Timestamp
    returns_dict: dict
    forward: pd.DataFrame
    seed: int

    def realized(self, horizon: int) -> pd.Series:
        """Simple return of every asset over the next ``horizon`` rows, the outcome.

        The daily returns compound: ``prod(1 + r) - 1``, the price ``horizon``
        rows after the origin over the origin's price, minus one.

        Raises
        ------
        ValueError
            If ``forward`` holds fewer than ``horizon`` rows.
        """
        if not 1 <= horizon <= len(self.forward):
            raise ValueError(
                f"A {horizon}-day outcome needs {horizon} forward returns; the "
                f"origin keeps {len(self.forward)}."
            )
        return (1 + self.forward.iloc[:horizon]).prod() - 1


def draw_trial(
    prices: pd.DataFrame,
    index: int,
    *,
    seed: int,
    n_assets: int,
    n_origins: int,
    origin_history: int,
    forward: int,
) -> Trial:
    """Draw a trial's assets and forecast origins without replacement.

    The look-back window is not a parameter, so runs that differ only in it
    draw the same trial.

    Parameters
    ----------
    prices : pd.DataFrame
        Price table (dates x assets), oldest row first.
    index : int
        Trial index.
    seed : int
        Experiment seed; with ``index`` it seeds the draw.
    n_assets : int
        Number of assets in the trial's universe.
    n_origins : int
        Number of forecast origins.
    origin_history : int
        Returns an origin needs before it; an origin needs ``origin_history``
        earlier price rows.
    forward : int
        Returns after the origin kept for scoring; an origin needs ``forward``
        later price rows.

    Returns
    -------
    Trial

    Raises
    ------
    ValueError
        If the table has fewer assets than ``n_assets`` or fewer eligible
        origins than ``n_origins``.
    """
    eligible = np.arange(origin_history, len(prices) - forward)
    if n_assets > prices.shape[1]:
        raise ValueError(
            f"A trial draws {n_assets} assets, but the price table has "
            f"{prices.shape[1]}."
        )
    if n_origins > len(eligible):
        raise ValueError(
            f"A trial draws {n_origins} origins, but only {len(eligible)} rows "
            f"have {origin_history} earlier and {forward} later rows."
        )
    rng = np.random.default_rng([seed, index])
    asset_positions = np.sort(rng.choice(prices.shape[1], n_assets, replace=False))
    origins = np.sort(rng.choice(eligible, n_origins, replace=False))
    return Trial(
        index=index,
        assets=tuple(str(prices.columns[i]) for i in asset_positions),
        origins=tuple(int(i) for i in origins),
    )


def scenario_seed(seed: int, trial: int, origin: int) -> int:
    """Scenario seed of one origin of one trial, identical for every arm."""
    return int(np.random.SeedSequence([seed, trial, origin]).generate_state(1)[0])


def origin_inputs(
    prices: pd.DataFrame,
    trial: Trial,
    origin: int,
    *,
    seed: int,
    look_back: int,
    forward: int,
) -> OriginInputs:
    """Build the look-back window and the returns after one origin.

    Parameters
    ----------
    prices : pd.DataFrame
        Price table the trial was drawn from.
    trial : Trial
        The trial.
    origin : int
        One of ``trial.origins``.
    seed : int
        Experiment seed.
    look_back : int
        Returns per look-back window.
    forward : int
        Returns after the origin.

    Returns
    -------
    OriginInputs

    Raises
    ------
    ValueError
        If ``calculate_returns`` dropped an asset (a missing price), if the
        window does not hold ``look_back`` returns, or if a price of the
        window or the forward rows is missing or not positive.
    """
    universe = prices[list(trial.assets)]
    returns_dict = calculate_returns(
        universe.iloc[origin - look_back : origin + 1],
        returns_compute_settings=RETURNS_SETTINGS,
    )
    window = returns_dict["returns"]
    if tuple(returns_dict["tickers"]) != trial.assets:
        dropped = sorted(set(trial.assets) - set(returns_dict["tickers"]))
        raise ValueError(
            f"Trial {trial.index} window ending {prices.index[origin]} has missing "
            f"prices for {dropped}."
        )
    if len(window) != look_back:
        raise ValueError(
            f"Trial {trial.index} window ending {prices.index[origin]} holds "
            f"{len(window)} returns, expected {look_back}."
        )
    if not (universe.iloc[origin - look_back : origin + forward + 1] > 0).all(
        axis=None
    ):
        raise ValueError(
            f"Trial {trial.index} window ending {prices.index[origin]} has a "
            f"missing or non-positive price."
        )
    return OriginInputs(
        origin=prices.index[origin],
        returns_dict=returns_dict,
        forward=universe.iloc[origin : origin + forward + 1].pct_change().iloc[1:],
        seed=scenario_seed(seed, trial.index, origin),
    )


# Separates the rebalancing draws of a trial from its asset and origin draws,
# so adding them changed none of the trials already drawn.
REBALANCING_STREAM = 0x52454241


@dataclass(frozen=True)
class TradingWindow:
    """A trial's rebalancing window: row positions of its first and last day."""

    start: int
    end: int


def draw_trading_window(
    prices: pd.DataFrame,
    index: int,
    *,
    seed: int,
    trading_days: int,
    look_back_days: int,
) -> TradingWindow:
    """Draw a trial's trading window uniformly among those that fit the data.

    A window needs ``look_back_days`` calendar days of history before its first
    day, for the first optimization and the risk-aversion calibration, and
    ``trading_days`` rows after it.

    Raises
    ------
    ValueError
        If no window of that length fits.
    """
    earliest = prices.index[0] + pd.Timedelta(days=look_back_days)
    first = int(prices.index.searchsorted(earliest))
    last = len(prices) - 1 - trading_days
    if last < first:
        raise ValueError(
            f"No {trading_days}-day trading window with {look_back_days} calendar "
            f"days of history fits {prices.index[0]:%Y-%m-%d} to "
            f"{prices.index[-1]:%Y-%m-%d}."
        )
    rng = np.random.default_rng([seed, index, REBALANCING_STREAM])
    start = int(rng.integers(first, last + 1))
    return TradingWindow(start=start, end=start + trading_days)


def rebalancing_seed(seed: int, trial: int) -> int:
    """Scenario seed of a trial's rebalancing runs, identical for every arm."""
    return int(
        np.random.SeedSequence([seed, trial, REBALANCING_STREAM]).generate_state(1)[0]
    )

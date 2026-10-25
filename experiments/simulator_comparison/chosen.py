# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Risk of the portfolios the rebalancing backtest chooses, read against their outcome.

An optimizer seeks out the portfolios whose risk its scenarios understate most,
so the random portfolios of ``PortfolioScores`` can miss the shortfall that
decisions feel. At every re-optimization, ``DecisionRecorder`` takes the chosen
portfolio's one-day log returns under the scenarios it was chosen on, in the
convention of ``PortfolioScores`` (``portfolio_log_returns``: simple returns
aggregated by weight, cash earning nothing on both sides), and records:

- ``var``: the value at risk at the optimizer's confidence, the loss its
  ``1 - confidence`` quantile marks, as ``PortfolioScores`` places it.
- ``cvar``: the conditional value at risk, the Rockafellar-Uryasev expression
  the optimizer minimizes, ``var + E[(loss - var)+] / (1 - confidence)``.
- ``mean`` and ``std``: the scenario mean and standard deviation.
- ``holdings``: the effective number of holdings, ``(sum |w|)^2 / sum w^2``
  over the asset weights, which is ``n`` for ``n`` positions of equal size
  whatever the cash weight and leverage, so it reads on the portfolio-size
  axis of ``PortfolioScores``.

``chosen_readings`` holds each decision's portfolio over the trading days until
the next decision, the last until the window ends, and counts the days whose
realized log return falls below ``-var`` and the squared standardized realized
returns ``((r - mean) / std)^2``. Over a run these give the breach rate, on
average ``1 - confidence`` when the scenarios state the chosen portfolios' risk
correctly, and the ratio of realized to scenario volatility, the root of the
mean squared standardized return.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_optimization.cvar_data import CvarData
from portfolio_optimization.portfolio import Portfolio

from .portfolios import portfolio_log_returns

# Decision dates are stored as ISO strings, which Parquet appends to rows read
# back from a file, unlike datetimes.
DATE_FORMAT = "%Y-%m-%d"
# Per-decision values of a run's row; the record first, then the realized readings.
DECISION_COLUMNS = (
    "chosen_dates",
    "chosen_var",
    "chosen_cvar",
    "chosen_mean",
    "chosen_std",
    "chosen_holdings",
    "chosen_days",
    "chosen_breaches",
    "chosen_z2",
)
# Per-run readings of a run's row, reported per arm.
CHOSEN_METRICS = (
    "chosen_breach_rate",
    "chosen_volatility_ratio",
    "chosen_effective_holdings",
)


@dataclass(frozen=True)
class Decision:
    """One re-optimization's chosen portfolio and its one-day scenario risk.

    ``var`` and ``cvar`` are losses, positive when the tail loses money;
    ``mean`` and ``std`` describe the scenario log returns.
    """

    date: pd.Timestamp
    tickers: tuple[str, ...]
    weights: np.ndarray
    var: float
    cvar: float
    mean: float
    std: float
    holdings: float


def effective_holdings(weights: np.ndarray) -> float:
    """``(sum |w|)^2 / sum w^2`` over asset weights: ``n`` for ``n`` equal positions.

    Raises
    ------
    ValueError
        If no asset carries weight.
    """
    squares = float(np.square(weights).sum())
    if squares == 0.0:
        raise ValueError("The portfolio holds no asset.")
    return float(np.abs(weights).sum() ** 2 / squares)


def scenario_risk(
    scenarios: np.ndarray, weights: np.ndarray, confidence: float
) -> dict[str, float]:
    """VaR, CVaR, mean, and standard deviation of one portfolio's scenario log returns.

    Parameters
    ----------
    scenarios : np.ndarray
        Equally likely asset log returns, shape ``(scenarios, n_assets)``.
    weights : np.ndarray
        Asset weights, shape ``(n_assets,)``.
    confidence : float
        CVaR level in (0, 1); the tail is the worst ``1 - confidence``.
    """
    returns = portfolio_log_returns(scenarios, weights[None, :])[0]
    tail = 1 - confidence
    var = -float(np.quantile(returns, tail))
    cvar = var + float(np.maximum(-returns - var, 0.0).mean()) / tail
    return {
        "var": var,
        "cvar": cvar,
        "mean": float(returns.mean()),
        "std": float(returns.std()),
    }


class DecisionRecorder:
    """Observer of ``rebalance_portfolio``: records every chosen portfolio's scenario risk.

    It reduces each decision's scenarios to the values of ``Decision`` at
    once, so a run keeps no scenario set beyond the one being optimized.

    Parameters
    ----------
    confidence : float
        CVaR level of the optimization.
    """

    def __init__(self, confidence: float):
        self.confidence = confidence
        self.decisions: list[Decision] = []

    def __call__(
        self, date: pd.Timestamp, cvar_data: CvarData, portfolio: Portfolio
    ) -> None:
        """Record the portfolio chosen on ``date`` from ``cvar_data``.

        Raises
        ------
        ValueError
            If the scenarios are not equally likely, or the scenario and
            portfolio asset counts differ.
        """
        p = np.asarray(cvar_data.p)
        if not np.allclose(p, 1 / len(p)):
            raise ValueError(
                "The chosen-portfolio risk reads equally likely scenarios; "
                "these carry unequal probabilities."
            )
        weights = np.asarray(portfolio.weights, dtype=float)
        scenarios = np.asarray(cvar_data.R, dtype=float).T
        if scenarios.shape[1] != len(weights) or len(portfolio.tickers) != len(weights):
            raise ValueError(
                f"Scenarios over {scenarios.shape[1]} assets cannot price a "
                f"portfolio of {len(weights)} weights over "
                f"{len(portfolio.tickers)} tickers."
            )
        self.decisions.append(
            Decision(
                date=pd.Timestamp(date),
                tickers=tuple(portfolio.tickers),
                weights=weights,
                holdings=effective_holdings(weights),
                **scenario_risk(scenarios, weights, self.confidence),
            )
        )


def chosen_readings(
    decisions: Sequence[Decision], realized: pd.DataFrame
) -> dict[str, object]:
    """Each decision's record and realized readings, and the run's readings.

    Decision ``k`` holds its portfolio on the dates of ``realized`` after its
    own date, up to and including the next decision's date; the last decision
    holds it to the last date. Every date must belong to one decision.

    Parameters
    ----------
    decisions : sequence of Decision
        The run's decisions in date order, the first on the window's first day.
    realized : pd.DataFrame
        Daily asset log returns of the trading window, dated by the day each
        return ends on, one column per ticker of the decisions.

    Returns
    -------
    dict
        The lists of ``DECISION_COLUMNS``, one entry per decision, and the
        floats of ``CHOSEN_METRICS``: the breach rate over every held day, the
        root mean squared standardized return, and the effective number of
        holdings averaged over the held days.

    Raises
    ------
    ValueError
        If there are no decisions, their dates do not increase, a return
        precedes the first decision, a decision holds no day, the tickers
        differ from the returns' columns, or a scenario spread is zero.
    """
    if not decisions:
        raise ValueError("The run recorded no decision.")
    dates = pd.DatetimeIndex([decision.date for decision in decisions])
    if not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError(f"Decision dates {list(dates)} do not increase.")
    if realized.index[0] <= dates[0]:
        raise ValueError(
            f"The returns start on {realized.index[0]}, not after the first "
            f"decision on {dates[0]}."
        )
    ends = [*dates[1:], realized.index[-1]]
    columns = {column: [] for column in DECISION_COLUMNS}
    for decision, end in zip(decisions, ends):
        if decision.tickers != tuple(realized.columns):
            raise ValueError(
                f"The decision of {decision.date} holds {decision.tickers}, the "
                f"returns cover {tuple(realized.columns)}."
            )
        if decision.std == 0.0:
            raise ValueError(f"The scenarios of {decision.date} have no spread.")
        held = realized.loc[(realized.index > decision.date) & (realized.index <= end)]
        if held.empty:
            raise ValueError(f"The decision of {decision.date} holds no trading day.")
        returns = portfolio_log_returns(held.to_numpy(), decision.weights[None, :])[0]
        z = (returns - decision.mean) / decision.std
        for column, value in (
            ("chosen_dates", decision.date.strftime(DATE_FORMAT)),
            ("chosen_var", decision.var),
            ("chosen_cvar", decision.cvar),
            ("chosen_mean", decision.mean),
            ("chosen_std", decision.std),
            ("chosen_holdings", decision.holdings),
            ("chosen_days", len(returns)),
            ("chosen_breaches", int((returns < -decision.var).sum())),
            ("chosen_z2", float(np.square(z).sum())),
        ):
            columns[column].append(value)
    days = np.asarray(columns["chosen_days"])
    if days.sum() != len(realized):
        raise ValueError(
            f"The decisions hold {days.sum()} of the window's {len(realized)} days."
        )
    return {
        **columns,
        "chosen_breach_rate": float(np.sum(columns["chosen_breaches"]) / days.sum()),
        "chosen_volatility_ratio": float(
            np.sqrt(np.sum(columns["chosen_z2"]) / days.sum())
        ),
        "chosen_effective_holdings": float(
            np.dot(columns["chosen_holdings"], days) / days.sum()
        ),
    }

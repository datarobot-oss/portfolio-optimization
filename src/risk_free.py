# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Risk-free rate that the cash position earns.

The rate is annual, either one number or a series of quotes by date. The
3-month US Treasury bill rate is the usual proxy: ``download_risk_free_rates``
saves its daily history, FRED series DTB3 (Federal Reserve Bank of St. Louis,
https://fred.stlouisfed.org/series/DTB3), and ``load_risk_free_rates`` reads
it back. ``cash_returns`` turns either form into the per-period return of
cash that the optimizer, the backtester, and the Sharpe and Sortino ratios use.
"""

from pathlib import Path

import numpy as np
import pandas as pd

FRED_DTB3_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
TRADING_DAYS_PER_YEAR = 252
RATE_COLUMN = "rate"
PERCENT = 100.0
# Longer than any run of market holidays, so only a series that ends early trips it.
MAX_QUOTE_AGE = pd.Timedelta(days=7)


def download_risk_free_rates(path: str | Path) -> pd.Series:
    """Download the daily 3-month Treasury bill rate and save it to ``path``.

    FRED quotes DTB3 in percent on a discount basis and leaves holidays empty;
    the saved CSV holds the quoted days only, as annual decimal rates in a
    ``rate`` column indexed by ``Date``.

    Returns
    -------
    pd.Series
        The saved rates, as ``load_risk_free_rates`` reads them.
    """
    raw = pd.read_csv(FRED_DTB3_URL, index_col=0, parse_dates=True, na_values=".")
    rates = (raw.iloc[:, 0].dropna() / PERCENT).rename(RATE_COLUMN)
    rates.index.name = "Date"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rates.to_csv(path)
    return load_risk_free_rates(path)


def load_risk_free_rates(path: str | Path) -> pd.Series:
    """Annual decimal risk-free rates by date from a ``download_risk_free_rates`` CSV.

    Raises
    ------
    ValueError
        If the file holds no rates, a rate is missing or not finite, or the
        dates are not strictly increasing.
    """
    rates = pd.read_csv(path, index_col=0, parse_dates=True)[RATE_COLUMN]
    if rates.empty:
        raise ValueError(f"{path} holds no risk-free rates.")
    if not np.isfinite(rates.to_numpy()).all():
        raise ValueError(f"{path} has a missing or non-finite risk-free rate.")
    if not rates.index.is_monotonic_increasing or not rates.index.is_unique:
        raise ValueError(f"The dates of {path} are not strictly increasing.")
    return rates


def period_returns(
    annual_rate: float | np.ndarray, return_type: str, freq: int = 1
) -> float | np.ndarray:
    """Return of cash over ``freq`` trading days at an annual decimal rate.

    The rate compounds over ``freq / TRADING_DAYS_PER_YEAR`` years, and the
    return is given in ``return_type`` units: ``LOG`` or ``LINEAR``.

    Raises
    ------
    NotImplementedError
        If ``return_type`` is neither ``LOG`` nor ``LINEAR``.
    """
    growth = (1 + annual_rate) ** (freq / TRADING_DAYS_PER_YEAR)
    if return_type == "LOG":
        return np.log(growth)
    if return_type == "LINEAR":
        return growth - 1
    raise NotImplementedError(
        f"Cash returns need LOG or LINEAR returns, not {return_type}."
    )


def cash_returns(
    risk_free_rate: float | pd.Series,
    dates: pd.DatetimeIndex,
    return_type: str,
    freq: int = 1,
) -> np.ndarray:
    """Return of cash over the period of ``freq`` trading days ending on each date.

    A period earns the annual rate quoted last before its end date, so a
    return never uses a rate quoted on its own last day; a scalar rate applies
    to every period. ``period_returns`` compounds the rate over the period.

    Parameters
    ----------
    risk_free_rate : float or pd.Series
        One annual decimal rate, or annual decimal rates indexed by quote date.
    dates : pd.DatetimeIndex
        End date of each period.
    return_type : str
        ``LOG`` or ``LINEAR``, as in ``ReturnsComputeSettings``.
    freq : int, default 1
        Trading days per period.

    Returns
    -------
    np.ndarray
        One cash return per date.

    Raises
    ------
    ValueError
        If a date has no rate quoted within ``MAX_QUOTE_AGE`` before it, as
        when the rates start after the dates or end before them.
    """
    if isinstance(risk_free_rate, pd.Series):
        before = dates - pd.Timedelta(days=1)
        quoted = pd.Series(risk_free_rate.index, index=risk_free_rate.index)
        age = before - pd.DatetimeIndex(quoted.asof(before))
        if age.isna().any() or (age > MAX_QUOTE_AGE).any():
            raise ValueError(
                f"The risk-free rates, quoted {risk_free_rate.index[0]:%Y-%m-%d} to "
                f"{risk_free_rate.index[-1]:%Y-%m-%d}, do not cover the periods "
                f"ending {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}."
            )
        annual = risk_free_rate.asof(before).to_numpy()
    else:
        annual = np.full(len(dates), float(risk_free_rate))
    return period_returns(annual, return_type, freq)

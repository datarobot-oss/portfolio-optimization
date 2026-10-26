# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Market regimes of the forecast origins: three origin features and their terciles.

An average over all origins can hide that a simulator gains in one market
regime and fails in another, so the report repeats its readings per tercile of
three features of each origin. They are computed from the returns the arms and
the scores saw at that origin (``origin_inputs``):

- ``window_correlation``: mean pairwise correlation of the look-back window.
- ``volatility_ratio``: each asset's volatility over the last
  ``TRAILING_STD_WINDOW`` returns of the window over its volatility over the
  whole window, median over assets; above 1 when volatility rose into the
  origin.
- ``correlation_change``: mean pairwise correlation of the forward returns
  minus ``window_correlation``. It looks ahead, so it explains readings and
  never selects a forecast.

Terciles are taken over the reported origins, and every arm of an origin falls
into the same regime, so the paired comparisons stay paired. Within a regime,
a trial's reading is its aggregate over its origins in that regime, and the
intervals resample trials as everywhere in the report.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    MIN_TRIALS,
    aggregate,
    comparisons,
    ranking,
    skill,
    skill_summary,
    summary,
)
from .portfolio_analysis import size_curves
from .scores import TRAILING_STD_WINDOW, pairwise_correlations
from .trials import OriginInputs, draw_trial, origin_inputs

FEATURES = ("window_correlation", "volatility_ratio", "correlation_change")
REGIMES = ("low", "middle", "high")
# Feature and regime of the readings over all origins.
OVERALL = "all"


def check_prices(prices_path: Path, identity: dict) -> None:
    """Refuse a price file other than the one the results were measured on.

    Raises
    ------
    ValueError
        If the SHA-256 of ``prices_path`` differs from the results' identity.
    """
    digest = hashlib.sha256(prices_path.read_bytes()).hexdigest()
    if digest != identity["prices"]["sha256"]:
        raise ValueError(
            f"{prices_path} is not the price file {identity['prices']['file']} the "
            f"results were measured on (its SHA-256 differs), so the origin "
            f"features cannot be recomputed from it."
        )


def _features(inputs: OriginInputs) -> dict[str, float]:
    window = inputs.returns_dict["returns"].to_numpy()
    forward = inputs.forward.to_numpy()
    correlation = pairwise_correlations(window, "look-back window").mean()
    trailing_std = window[-TRAILING_STD_WINDOW:].std(axis=0)
    return {
        "window_correlation": float(correlation),
        "volatility_ratio": float(np.median(trailing_std / window.std(axis=0))),
        "correlation_change": float(
            pairwise_correlations(forward, "forward").mean() - correlation
        ),
    }


def origin_features(
    prices: pd.DataFrame, frame: pd.DataFrame, experiment: dict
) -> pd.DataFrame:
    """The ``FEATURES`` of every (trial, origin) in ``frame``.

    Each trial is drawn again from ``experiment`` (the results' identity entry),
    so the features use the assets and windows the arms saw.

    Parameters
    ----------
    prices : pd.DataFrame
        The price table the results were measured on.
    frame : pd.DataFrame
        Results rows; only ``trial`` and ``origin`` are read.
    experiment : dict
        ``seed``, ``assets``, ``origins``, ``look_back``, ``origin_history``,
        and ``forward`` of the run.

    Returns
    -------
    pd.DataFrame
        One row per (trial, origin), indexed by both, one column per feature.

    Raises
    ------
    ValueError
        If an origin of ``frame`` is not one its trial draws from ``prices``,
        which means the results came from other prices or parameters.
    """
    rows = []
    for index, dates in frame.groupby("trial")["origin"].unique().items():
        trial = draw_trial(
            prices,
            int(index),
            seed=experiment["seed"],
            n_assets=experiment["assets"],
            n_origins=experiment["origins"],
            origin_history=experiment["origin_history"],
            forward=experiment["forward"],
        )
        positions = pd.Series(trial.origins, index=prices.index[list(trial.origins)])
        dates = pd.DatetimeIndex(dates).sort_values()
        unknown = dates.difference(positions.index)
        if len(unknown):
            raise ValueError(
                f"Trial {index} has origins {list(unknown)} that it does not draw "
                f"from these prices and parameters."
            )
        for date in dates:
            inputs = origin_inputs(
                prices,
                trial,
                int(positions[date]),
                seed=experiment["seed"],
                look_back=experiment["look_back"],
                forward=experiment["forward"],
            )
            rows.append({"trial": index, "origin": date, **_features(inputs)})
    return pd.DataFrame(rows).set_index(["trial", "origin"])


def terciles(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each origin's regime per feature, and each regime's bounds and size.

    Returns
    -------
    tuple
        The regimes, indexed like ``features`` with one ``REGIMES`` label per
        feature column, and a table with one row per (feature, regime): its
        ``low`` and ``high`` feature bounds and its numbers of origins and trials.

    Raises
    ------
    ValueError
        If tied feature values make two tercile bounds equal.
    """
    labels = {}
    rows = []
    for feature in FEATURES:
        regime, bounds = pd.qcut(
            features[feature], len(REGIMES), labels=REGIMES, retbins=True
        )
        labels[feature] = regime.astype(str)
        for name, low, high in zip(REGIMES, bounds[:-1], bounds[1:]):
            members = regime.index[regime == name]
            rows.append(
                {
                    "feature": feature,
                    "regime": name,
                    "low": float(low),
                    "high": float(high),
                    "origins": len(members),
                    "trials": members.get_level_values("trial").nunique(),
                }
            )
    return pd.DataFrame(labels, index=features.index), pd.DataFrame(rows)


def _scores(per_trial: pd.DataFrame) -> pd.DataFrame:
    scores = summary(per_trial)
    return scores[scores["metric"] != "seconds"]


def _skill(per_trial: pd.DataFrame) -> pd.DataFrame:
    return skill_summary(skill(per_trial))


# How each reading repeated per regime is read off a cell's per-trial table.
READINGS = {
    "summary": _scores,
    "comparisons": comparisons,
    "skill": _skill,
    "ranking": ranking,
    "portfolio_sizes": size_curves,
}


def regime_readings(
    frame: pd.DataFrame,
    regimes: pd.DataFrame,
    readings: Sequence[str] = tuple(READINGS),
) -> dict[str, pd.DataFrame]:
    """The report's readings over all origins and in every regime.

    For all origins and for each (feature, regime), the rows of those origins
    are aggregated per trial and read by every reading of ``readings``:
    ``summary`` (without timing), ``comparisons``, ``skill_summary``,
    ``ranking``, and ``size_curves``. A caller that needs only some of them
    names those, since the Wilcoxon tests of ``comparisons`` cost most of the
    time. The readings over all origins equal those of the report's overall
    tables.

    Returns
    -------
    dict
        One table per name in ``readings``, each with leading ``feature`` and
        ``regime`` columns; the readings over all origins carry ``OVERALL`` in both.

    Raises
    ------
    ValueError
        If a name of ``readings`` is not in ``READINGS``, an origin of
        ``frame`` has no regime, or a regime holds fewer than ``MIN_TRIALS``
        trials.
    """
    unknown = sorted(set(readings) - set(READINGS))
    if unknown:
        raise ValueError(f"Unknown regime readings {unknown}; known: {list(READINGS)}.")
    keys = pd.MultiIndex.from_frame(frame[["trial", "origin"]])
    cells = [(OVERALL, OVERALL, np.ones(len(frame), dtype=bool))]
    for feature in FEATURES:
        of_row = regimes[feature].reindex(keys)
        if of_row.isna().any():
            raise ValueError(f"Some origins have no {feature} regime.")
        cells += [(feature, name, (of_row == name).to_numpy()) for name in REGIMES]
    tables = {name: [] for name in readings}
    for feature, name, mask in cells:
        per_trial = aggregate(frame[mask])
        trials = per_trial.index.get_level_values("trial").nunique()
        if trials < MIN_TRIALS:
            raise ValueError(
                f"The {name} {feature} regime holds {trials} trials; its readings "
                f"need at least {MIN_TRIALS}."
            )
        for reading in readings:
            tables[reading].append(
                READINGS[reading](per_trial)
                .assign(feature=feature, regime=name)
                .set_index(["feature", "regime"])
            )
    return {name: pd.concat(parts).reset_index() for name, parts in tables.items()}

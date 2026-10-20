# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration of a simulator-comparison run, read from a YAML file.

One file drives the run and its report: the report reads the results from the
file's ``output``. Relative paths are resolved against the working directory,
which is the repository root when the run starts from the Taskfile. Every
field must be given in the YAML file, so a config file alone says what ran.
Services and GPUs are not configured here: the arms read ``.env`` through the
library's defaults, as every user of the library does.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .arms import CONFIG_ARM_NAMES, JOINTFM_ARM


def _known_unique_arms(arms: tuple[str, ...]) -> tuple[str, ...]:
    unknown = [arm for arm in arms if arm not in CONFIG_ARM_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown arms {unknown}; choose from {list(CONFIG_ARM_NAMES)}."
        )
    if len(set(arms)) != len(arms):
        raise ValueError("Each arm may run only once.")
    return arms


class CvarConfig(BaseModel):
    """CVaR optimization constraints of every rebalancing run (``CvarParameters``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    w_min: float = Field(
        description="Lower bound of each asset weight; below 0 allows shorts."
    )
    w_max: float = Field(description="Upper bound of each asset weight.")
    c_min: float = Field(description="Lower bound of the cash weight.")
    c_max: float = Field(description="Upper bound of the cash weight.")
    L_tar: float = Field(gt=0, description="Upper bound of gross exposure (leverage).")
    T_tar: float = Field(
        gt=0, description="Upper bound of turnover per re-optimization."
    )
    confidence: float = Field(
        gt=0, lt=1, description="Confidence level of the CVaR the optimizer limits."
    )
    risk_aversion: float = Field(
        gt=0,
        description="Base risk aversion. Each trial scales it once by the best "
        "single-asset return/CVaR ratio of the history before its trading window, "
        "and the runs keep that value (no per-window scaling).",
    )


class RebalancingConfig(BaseModel):
    """Rebalancing backtest of every trial's universe, one run per arm."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output: Path = Field(
        description="Results Parquet file of the rebalancing runs, resumed like the "
        "scenario results; its report goes to <output stem>_report beside it."
    )
    arms: tuple[str, ...] = Field(
        min_length=1,
        description=f"Simulators to backtest, in this order: {', '.join(CONFIG_ARM_NAMES)}. "
        f"The device changes only speed, not the portfolios, so one arm per "
        f"simulator suffices; {JOINTFM_ARM} reaches the service .env selects.",
    )
    trials: int = Field(
        gt=0,
        description="Trials 0 .. trials - 1 are backtested. Trial i trades the "
        "assets of scenario trial i, drawn from the experiment seed and i alone, so "
        "the count is independent of the scenario comparison's trials and the "
        "trials both run share their universes; raising it extends a finished run.",
    )
    trading_days: int = Field(
        gt=0,
        description="Length in trading days of each trial's trading window, whose "
        "start each trial draws at random; a multiple of look_forward_window so "
        "every step is complete.",
    )
    look_forward_window: int = Field(
        gt=0, description="Trading days per step: the holding period between checks."
    )
    look_back_days: int = Field(
        gt=0,
        description="Calendar days of history each re-optimization fits; "
        "rebalance_portfolio counts this window in calendar, not trading, days.",
    )
    pct_change_threshold: float = Field(
        description="Percentage-change trigger: re-optimize when a step's return "
        "falls below this fraction (negative, e.g. -0.005 for a 0.5 % loss)."
    )
    transaction_cost: float = Field(
        ge=0, description="Transaction cost as a fraction of turnover."
    )
    risk_free_rate: float = Field(
        description="Annual risk-free rate the Sharpe and Sortino ratios subtract, "
        "compounded to a daily rate."
    )
    solver: Literal["CUOPT", "CLARABEL"] = Field(
        description="CVXPY solver of every optimization: CUOPT (GPU, PDLP) for runs, "
        "CLARABEL (CPU) for hosts without cuOpt."
    )
    cvar: CvarConfig = Field(description="CVaR optimization constraints.")

    @field_validator("arms")
    @classmethod
    def known_unique_arms(cls, arms: tuple[str, ...]) -> tuple[str, ...]:
        return _known_unique_arms(arms)

    @model_validator(mode="after")
    def whole_steps(self) -> "RebalancingConfig":
        if self.trading_days % self.look_forward_window:
            raise ValueError(
                f"trading_days ({self.trading_days}) must be a multiple of "
                f"look_forward_window ({self.look_forward_window}) so every step "
                f"is complete."
            )
        return self

    @property
    def report_dir(self) -> Path:
        return self.output.with_name(f"{self.output.stem}_report")

    @property
    def universe_dir(self) -> Path:
        """Per-trial price CSVs, which rebalance_portfolio reads by path."""
        return self.output.with_name(f"{self.output.stem}_universes")


class ExperimentConfig(BaseModel):
    """Parameters of one simulator-comparison run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prices: Path = Field(
        description="Price CSV, dates in the first column and one column per "
        "asset, oldest row first; the trials draw their assets and origins from it."
    )
    output: Path = Field(
        description="Results Parquet file. An existing file is resumed: rows "
        "already in it are skipped, and a file measured under other conditions "
        "is refused."
    )
    trials: int = Field(
        gt=0,
        description="Trials 0 .. trials - 1 are run. Each trial is one seeded "
        "draw of assets and origins shared by every arm; raising the count "
        "extends a finished run without changing its trials.",
    )
    origins: int = Field(gt=0, description="Forecast origins per trial.")
    assets: int = Field(
        gt=0,
        description="Assets per trial; must fit the JointFM services' max_series.",
    )
    look_back: int = Field(
        gt=0,
        description="Returns per look-back window, the history every arm "
        "receives; the scores need at least 63 for the trailing volatility.",
    )
    forward: int = Field(
        gt=1,
        description="Returns after each origin kept for the forward "
        "correlations; only the scores read them, no arm sees them.",
    )
    num_scen: int = Field(
        gt=0,
        description="Scenarios per generating arm; no_fit uses the look-back "
        "window's returns instead.",
    )
    seed: int = Field(
        description="Experiment seed of the trial draws and the scenario seeds."
    )
    arms: tuple[str, ...] = Field(
        min_length=1,
        description=f"Arms to run, in this order, each at most once: "
        f"{', '.join(CONFIG_ARM_NAMES)}. {JOINTFM_ARM} reaches the service .env "
        f"selects and runs as jointfm-cpu or jointfm-gpu by that service's device.",
    )

    rebalancing: RebalancingConfig = Field(
        description="Rebalancing backtest of the same trials' universes, run by "
        "its own task after or beside the scenario comparison."
    )

    @field_validator("arms")
    @classmethod
    def known_unique_arms(cls, arms: tuple[str, ...]) -> tuple[str, ...]:
        return _known_unique_arms(arms)

    @property
    def report_dir(self) -> Path:
        """Where the report writes: ``<output stem>_report`` beside the results."""
        return self.output.with_name(f"{self.output.stem}_report")


def load_config(path: Path) -> ExperimentConfig:
    """Read and validate a YAML configuration file.

    Raises
    ------
    pydantic.ValidationError
        If a field is missing, unknown, or invalid.
    """
    with open(path) as handle:
        return ExperimentConfig.model_validate(yaml.safe_load(handle))

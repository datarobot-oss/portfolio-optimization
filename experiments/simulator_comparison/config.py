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

from .arms import (
    CONFIG_ARM_NAMES,
    HISTORICAL_MEAN_ARM,
    INDEPENDENT_ARM,
    JOINTFM_ARM,
    SWAP_DONORS,
    WINDOW_ARM,
    check_donors,
)


def _known_unique_arms(arms: tuple[str, ...]) -> tuple[str, ...]:
    unknown = [arm for arm in arms if arm not in CONFIG_ARM_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown arms {unknown}; choose from {list(CONFIG_ARM_NAMES)}."
        )
    if len(set(arms)) != len(arms):
        raise ValueError("Each arm may run only once.")
    check_donors(arms)
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
    scenario_results: Path = Field(
        description="Scenario-comparison results file over whose next-day curves "
        "of random long-only and long-short portfolios the report draws the risk "
        "of the chosen portfolios: the config's own output, or, for a twin that "
        "runs only the backtest, the output of the config it is twin to. It must "
        "hold every backtested trial complete, on the same prices, universes, and "
        "weight, cash, and leverage bounds, so the scenario comparison runs "
        "before the report. Only the report reads it, so it is not part of the "
        "rebalancing results' identity."
    )
    arms: tuple[str, ...] = Field(
        min_length=1,
        description=f"Simulators to backtest, in this order: {', '.join(CONFIG_ARM_NAMES)}. "
        f"The device changes only speed, not the portfolios, so one arm per "
        f"simulator suffices; {JOINTFM_ARM} reaches the service .env selects. "
        f"A swap-* arm is a dependence swap cell and needs its two donors listed too. "
        f"{HISTORICAL_MEAN_ARM} optimizes on {JOINTFM_ARM}'s scenarios with the "
        f"window's mean as expected returns, so {JOINTFM_ARM} against it isolates "
        f"the expected returns and it against {WINDOW_ARM} the scenarios; it needs "
        f"{JOINTFM_ARM} listed too. {INDEPENDENT_ARM}, the positive control, "
        f"optimizes on the window's returns with each asset's rows permuted "
        f"independently, which keeps every asset's returns and mean and removes "
        f"all dependence; it needs {WINDOW_ARM} listed too.",
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
    trigger: Literal["every_step", "pct_change"] = Field(
        description="When the portfolio is re-optimized after a step: every_step "
        "re-optimizes after every look_forward_window, a fixed calendar that gives "
        "every arm the same decision dates; pct_change re-optimizes only when the "
        "step's return, or the sum of the losing steps since the last "
        "re-optimization, falls below pct_change_threshold."
    )
    pct_change_threshold: float = Field(
        description="Loss threshold of the pct_change trigger, a negative fraction "
        "(e.g. -0.005 for a 0.5 % loss); every_step ignores it."
    )
    transaction_cost: float = Field(
        ge=0, description="Transaction cost as a fraction of turnover."
    )
    risk_free_rates: Path = Field(
        description="CSV of annual risk-free rates by quote date, as "
        "`task data:risk-free-rates` downloads them (the 3-month Treasury bill "
        "rate, FRED DTB3). Cash earns them in every optimization and backtest, "
        "and the Sharpe and Sortino ratios subtract them; each period earns the "
        "rate quoted before it ends, compounded over 252 trading days a year."
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


class UniverseSizesConfig(BaseModel):
    """Rebalancing backtest of nested asset subsets of every trial's universe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output: Path = Field(
        description="Results Parquet file of the universe-size runs, one row per "
        "trial, size, and arm, resumed like the rebalancing results; the blog "
        "reads it. Only a base config's task experiments:universe-sizes writes "
        "it, so a twin names a file of its own that nothing runs."
    )
    arms: tuple[str, ...] = Field(
        min_length=1,
        description=f"Simulators to backtest at every size, named as in "
        f"rebalancing.arms; the blog draws each against {WINDOW_ARM}, so it must "
        f"be listed.",
    )
    sizes: tuple[int, ...] = Field(
        min_length=1,
        description="Assets per universe, strictly increasing and at most assets. "
        "Each trial fixes one random order of its assets, and a size trades the "
        "first ones of it, so every smaller universe is a subset of every larger "
        "one; size assets trades exactly the rebalancing backtest's universe. "
        "Every size of a trial shares its trading window, scenario seed, and "
        "rebalancing settings, while its risk aversion is calibrated on its own "
        "assets. Each size adds one run per trial and arm, at rebalancing.trials "
        "trials.",
    )

    @field_validator("arms")
    @classmethod
    def known_unique_arms(cls, arms: tuple[str, ...]) -> tuple[str, ...]:
        if WINDOW_ARM not in arms:
            raise ValueError(
                f"universe_sizes.arms must list {WINDOW_ARM}, which the blog "
                f"reads every simulator against."
            )
        return _known_unique_arms(arms)

    @field_validator("sizes")
    @classmethod
    def increasing_sizes(cls, sizes: tuple[int, ...]) -> tuple[int, ...]:
        if sizes[0] < 1 or any(b <= a for a, b in zip(sizes, sizes[1:])):
            raise ValueError(
                f"universe_sizes.sizes {list(sizes)} must be positive and strictly "
                f"increasing."
            )
        return sizes

    @property
    def universe_dir(self) -> Path:
        """Per-trial, per-size price CSVs, which rebalance_portfolio reads by path."""
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
    origin_history: int = Field(
        gt=0,
        description="Returns an origin needs before it to be drawn, at least "
        "look_back. Only the trial draw reads it, so configs that differ only in "
        "look_back draw the same origins and scenario seeds and their results "
        "pair trial by trial (paired.py, setting look_back).",
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
        f"selects and runs as jointfm-cpu or jointfm-gpu by that service's device. "
        f"A swap-<marginals>-<dependence> arm is a dependence swap cell: its marginal "
        f"donor's scenario values per asset in its dependence donor's rank order; "
        f"both donors (window = no_fit-cpu) must be listed too. "
        f"{INDEPENDENT_ARM} is the window's scenarios at every horizon with each "
        f"asset's rows permuted independently, and needs {WINDOW_ARM} listed too. "
        f"{HISTORICAL_MEAN_ARM} runs only in rebalancing.arms.",
    )
    portfolio_sizes: tuple[int, ...] = Field(
        min_length=1,
        description="Numbers of assets the scored portfolios hold, strictly "
        "increasing and at most assets. At every origin each size gets random "
        "asset subsets carrying an equal-weight portfolio and, where "
        "rebalancing.cvar's weight, cash, and leverage bounds allow one at that "
        "size, a random long-only and a random long-short portfolio. Size 1 reads "
        "the per-asset distributions alone; larger sizes read the dependence ever "
        "more.",
    )
    horizons: tuple[int, ...] = Field(
        min_length=1,
        description="Trading days after each origin at which every arm is "
        "scored, strictly increasing, starting at 1 (the next day, which every "
        "single-horizon reading of the report shows), and at most forward. "
        "JointFM forecasts all of them in one request; every other arm's h-day "
        "scenario is the sum of h independent one-day draws of its own "
        "simulator. Each horizon adds one results row per (trial, origin, arm).",
    )

    rebalancing: RebalancingConfig = Field(
        description="Rebalancing backtest of the same trials' universes, run by "
        "its own task after or beside the scenario comparison."
    )
    universe_sizes: UniverseSizesConfig = Field(
        description="Rebalancing backtest of nested asset subsets of the same "
        "trials' universes over the rebalancing settings, run by task "
        "experiments:universe-sizes for a base config; a twin carries it unused."
    )

    @field_validator("arms")
    @classmethod
    def known_unique_arms(cls, arms: tuple[str, ...]) -> tuple[str, ...]:
        if HISTORICAL_MEAN_ARM in arms:
            raise ValueError(
                f"{HISTORICAL_MEAN_ARM} runs only in rebalancing.arms: its scenarios "
                f"are {JOINTFM_ARM}'s and its expected returns the window's, which "
                f"{WINDOW_ARM} already reads."
            )
        return _known_unique_arms(arms)

    @model_validator(mode="after")
    def portfolio_sizes_fit(self) -> "ExperimentConfig":
        sizes = self.portfolio_sizes
        if sizes[0] < 1 or any(b <= a for a, b in zip(sizes, sizes[1:])):
            raise ValueError(
                f"portfolio_sizes {list(sizes)} must be positive and strictly "
                f"increasing."
            )
        if sizes[-1] > self.assets:
            raise ValueError(
                f"portfolio_sizes {list(sizes)} exceed the {self.assets} assets of "
                f"a trial."
            )
        return self

    @model_validator(mode="after")
    def universe_sizes_fit(self) -> "ExperimentConfig":
        sizes = self.universe_sizes.sizes
        if sizes[-1] > self.assets:
            raise ValueError(
                f"universe_sizes.sizes {list(sizes)} exceed the {self.assets} assets "
                f"of a trial."
            )
        return self

    @model_validator(mode="after")
    def origins_hold_the_window(self) -> "ExperimentConfig":
        if self.origin_history < self.look_back:
            raise ValueError(
                f"origin_history {self.origin_history} must be at least look_back "
                f"{self.look_back}, or an origin could lack its window."
            )
        return self

    @model_validator(mode="after")
    def horizons_fit(self) -> "ExperimentConfig":
        horizons = self.horizons
        if horizons[0] != 1 or any(b <= a for a, b in zip(horizons, horizons[1:])):
            raise ValueError(
                f"horizons {list(horizons)} must start at 1 and be strictly increasing."
            )
        if horizons[-1] > self.forward:
            raise ValueError(
                f"horizons {list(horizons)} exceed the {self.forward} forward "
                f"returns that score them."
            )
        return self

    @model_validator(mode="after")
    def swap_cells_hold_the_window(self) -> "ExperimentConfig":
        """A swap cell stretches the window's rows to ``num_scen`` and never drops one.

        The rebalancing window spans ``look_back_days`` calendar days, which
        bound its trading rows from above.
        """
        windows = {
            "look_back": (self.arms, self.look_back),
            "rebalancing.look_back_days": (
                (*self.rebalancing.arms, *self.universe_sizes.arms),
                self.rebalancing.look_back_days,
            ),
        }
        for key, (arms, rows) in windows.items():
            if any(arm in SWAP_DONORS for arm in arms) and self.num_scen < rows:
                raise ValueError(
                    f"Swap cells stretch the window's rows to num_scen, so "
                    f"num_scen {self.num_scen} must be at least {key} {rows}."
                )
        return self

    @property
    def report_dir(self) -> Path:
        """Where the report writes: ``<output stem>_report`` beside the results."""
        return self.output.with_name(f"{self.output.stem}_report")

    @property
    def blog_dir(self) -> Path:
        """Where the blog figures go: ``<output stem>_blog`` beside the results."""
        return self.output.with_name(f"{self.output.stem}_blog")


def load_config(path: Path) -> ExperimentConfig:
    """Read and validate a YAML configuration file.

    Raises
    ------
    pydantic.ValidationError
        If a field is missing, unknown, or invalid.
    """
    with open(path) as handle:
        return ExperimentConfig.model_validate(yaml.safe_load(handle))

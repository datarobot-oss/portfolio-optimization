# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebalancing backtest of each trial's universe, one run per simulator.

Each trial's 100 assets (the draw the scenario comparison uses) trade over a
window of ``trading_days`` that the trial draws at random, through the
library's ``rebalance_portfolio``: after every ``look_forward_window`` trading
days the configured ``trigger`` decides whether the portfolio is re-optimized,
always (``every_step``) or after losses beyond ``pct_change_threshold``
(``pct_change``). Every arm of a trial gets
the same window, CVaR constraints, risk aversion, and scenario seed, so arms
differ only in the ``CvarData`` their simulator generates: its scenarios and
its expected returns.

The risk aversion is the configured base scaled once per trial by the best
single-asset return/CVaR ratio of the history before the window, as the
rebalancing notebook does, and kept for every optimization of the trial.

Cash earns the dated risk-free rates of ``risk_free_rates``: every optimization
credits it with the rate quoted on its date, the backtest accrues it day by
day, and the Sharpe and Sortino ratios subtract the same daily rates.

Every return is a simple return (``RETURNS_SETTINGS``), so the optimizer's
portfolio return ``R @ w`` is exact, and portfolio values follow the library's
backtester as ``cumprod(1 + R @ w)``, which carries the full loss of a short
position.

Every optimization also records the chosen portfolio's one-day scenario risk
(``chosen.DecisionRecorder``), and the run reads it against the realized
returns of the days that portfolio is held (``chosen.chosen_readings``).
"""

import argparse
import copy
import hashlib
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from portfolio_optimization import cvar_utils, risk_free, utils
from portfolio_optimization.cvar_parameters import CvarParameters
from portfolio_optimization.gpu import apply_gpu_selection
from portfolio_optimization.settings import ApiSettings, ScenarioGenerationSettings

from .arms import (
    AnyArm,
    HistoricalMeanArm,
    IndependentArm,
    SwapArm,
    check_same_checkpoint,
    historical_mean_settings,
    jointfm_calls,
    resolve_arms,
    scenario_settings,
    verify_jointfm_service,
)
from .chosen import CHOSEN_METRICS, DecisionRecorder, chosen_readings
from .config import ExperimentConfig, RebalancingConfig, load_config
from .independence import IndependentScenarioGenerator
from .runner import (
    PROGRESS_INTERVAL_LOG,
    PROGRESS_INTERVAL_TERMINAL,
    host_metadata,
    load_prices,
    package_versions,
    run_identity,
)
from .swaps import SwapScenarioGenerator
from .table import RunMetadata, merge_identity, read_results, write_results
from .trials import (
    RETURNS_SETTINGS,
    TradingWindow,
    Trial,
    draw_trading_window,
    draw_trial,
    rebalancing_seed,
)

KEY_COLUMNS = ("trial", "arm")
TRADING_DAYS_PER_YEAR = risk_free.TRADING_DAYS_PER_YEAR
# Per-run readings; step_returns holds the per-step values behind the first two,
# the chosen_* lists of chosen.DECISION_COLUMNS the decisions behind CHOSEN_METRICS.
METRICS = (
    "mean_step_return",
    "step_return_std",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "turnover",
    "rebalances",
    "buy_and_hold_return",
    *CHOSEN_METRICS,
)


def solver_settings(solver: str) -> dict:
    """CVXPY solver settings of every optimization."""
    import cvxpy as cp

    if solver == "CUOPT":
        return {"solver": cp.CUOPT, "verbose": False, "solver_method": "PDLP"}
    return {"solver": cp.CLARABEL, "verbose": False}


def write_universe(prices: pd.DataFrame, trial: Trial, directory: Path) -> Path:
    """Write a trial's price columns to the CSV ``rebalance_portfolio`` reads."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"trial_{trial.index:04d}.csv"
    prices[list(trial.assets)].to_csv(path)
    return path


def calibrated_cvar_params(
    universe: pd.DataFrame, window: TradingWindow, config: RebalancingConfig
) -> CvarParameters:
    """The trial's CVaR parameters with its risk aversion calibrated once.

    Scales the base risk aversion by the best single-asset return/CVaR ratio of
    the universe's history up to the window's first day, as the rebalancing
    notebook does, so no data from the trading window enters it.
    """
    from portfolio_optimization import cvar_optimizer

    params = CvarParameters(**config.cvar.model_dump(), cvar_limit=None)
    history = utils.calculate_returns(
        universe,
        {
            "name": "risk-aversion calibration",
            "range": (
                universe.index[0].strftime("%Y-%m-%d"),
                universe.index[window.start].strftime("%Y-%m-%d"),
            ),
        },
        RETURNS_SETTINGS,
    )
    calibration = cvar_utils.generate_cvar_data(
        history, ScenarioGenerationSettings(fit_type="no_fit")
    )
    problem = cvar_optimizer.CVaR(returns_dict=calibration, cvar_params=params)
    params.update_risk_aversion(problem.params.risk_aversion)
    return params


def backtest_readings(
    results: pd.DataFrame,
    value: pd.Series,
    buy_and_hold: pd.Series,
    steps: int,
    risk_free_rates: pd.Series,
) -> dict:
    """Readings of one rebalancing run.

    ``value`` is the daily portfolio value, 1 on the first day; all returns are
    simple returns of it. A step's return is the value change from its first
    day to the next step's first day (the last step ends on the window's last
    day), so it includes the transaction cost paid when that step's portfolio
    was set up. The annualized readings use ``TRADING_DAYS_PER_YEAR`` and the
    daily returns: annualized return compounds the total return over the
    window's length; Sharpe and Sortino divide the mean daily return in excess
    of the day's risk-free rate (``risk_free.cash_returns``) by the standard
    deviation of daily returns and by their downside deviation (root mean
    square of the negative excess returns), scaled by the square root of the
    trading days per year; Calmar divides annualized return by maximum
    drawdown.

    Raises
    ------
    ValueError
        If the run did not cover ``steps`` complete steps.
    """
    if len(results) != steps:
        raise ValueError(f"The run covered {len(results)} steps, expected {steps}.")
    boundaries = [*results.index, value.index[-1]]
    at = value.reindex(boundaries).to_numpy()
    if not np.isfinite(at).all():
        raise ValueError("A step boundary has no portfolio value.")
    step_returns = at[1:] / at[:-1] - 1
    daily_series = value.pct_change().dropna()
    daily = daily_series.to_numpy()
    total_return = float(value.iloc[-1] / value.iloc[0] - 1)
    annualized_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(daily)) - 1
    excess = daily - risk_free.cash_returns(
        risk_free_rates, pd.DatetimeIndex(daily_series.index), "LINEAR"
    )
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    max_drawdown = float((1 - value / value.cummax()).max())
    per_year = np.sqrt(TRADING_DAYS_PER_YEAR)
    portfolios = list(results["optimal_portfolio"])
    turnover = sum(
        float(np.abs(new.weights - old.weights).sum() + abs(new.cash - old.cash))
        for old, new in zip(portfolios, portfolios[1:])
        if new is not old
    )
    return {
        "mean_step_return": float(step_returns.mean()),
        "step_return_std": float(step_returns.std(ddof=1)),
        "total_return": total_return,
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(daily.std(ddof=1) * per_year),
        "sharpe_ratio": float(excess.mean() / daily.std(ddof=1) * per_year),
        "sortino_ratio": float(excess.mean() / downside * per_year),
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(annualized_return / max_drawdown),
        "turnover": turnover,
        "rebalances": int(results["re_optimized"].sum()),
        "buy_and_hold_return": float(buy_and_hold.iloc[-1] / buy_and_hold.iloc[0] - 1),
        "step_returns": step_returns.tolist(),
    }


def backtest_scenarios(
    arm: AnyArm, *, num_scen: int, seed: int
) -> tuple[ScenarioGenerationSettings, Callable[..., dict]]:
    """Settings and scenario generator that make ``rebalance_portfolio`` run ``arm``.

    A simulator runs through the library's ``generate_cvar_data``. A swap cell
    runs on its marginal donor's settings, so it draws the marginal donor's
    scenarios and keeps its expected returns, while ``SwapScenarioGenerator``
    reorders them into the dependence donor's ranks; both donors share the
    seed their own arms run with, so they draw exactly those arms' scenarios.
    The historical-mean arm runs its JointFM donor's settings with
    ``expected_return="historical"`` under the same seed, so it optimizes on
    the donor's scenarios with the window's mean. The independence control
    runs the window's settings, and ``IndependentScenarioGenerator`` permutes
    each asset's rows.
    """
    if isinstance(arm, HistoricalMeanArm):
        return historical_mean_settings(
            arm, num_scen=num_scen, seed=seed
        ), cvar_utils.generate_cvar_data
    if isinstance(arm, IndependentArm):
        return scenario_settings(
            arm.window, num_scen=num_scen, seed=seed
        ), IndependentScenarioGenerator()
    if isinstance(arm, SwapArm):
        return scenario_settings(
            arm.marginals, num_scen=num_scen, seed=seed
        ), SwapScenarioGenerator(
            scenario_settings(arm.dependence, num_scen=num_scen, seed=seed), num_scen
        )
    return scenario_settings(
        arm, num_scen=num_scen, seed=seed
    ), cvar_utils.generate_cvar_data


def run_backtest(
    arm: AnyArm,
    trial: Trial,
    window: TradingWindow,
    universe_path: Path,
    params: CvarParameters,
    config: ExperimentConfig,
    risk_free_rates: pd.Series,
) -> dict:
    """One rebalancing run of ``arm`` on a trial's universe and window."""
    from portfolio_optimization import rebalance

    rebalancing = config.rebalancing
    dates = load_prices(universe_path).index
    settings, generator = backtest_scenarios(
        arm,
        num_scen=config.num_scen,
        seed=rebalancing_seed(config.seed, trial.index),
    )
    recorder = DecisionRecorder(params.confidence)
    start = time.perf_counter()
    rebalancer = rebalance.rebalance_portfolio(
        dataset_directory=str(universe_path),
        returns_compute_settings=RETURNS_SETTINGS,
        scenario_generation_settings=settings,
        scenario_generator=generator,
        trading_start=dates[window.start].strftime("%Y-%m-%d"),
        trading_end=dates[window.end].strftime("%Y-%m-%d"),
        look_forward_window=rebalancing.look_forward_window,
        look_back_window=rebalancing.look_back_days,
        # Each run gets its own copy, so no run can change another's constraints.
        cvar_params=copy.deepcopy(params),
        solver_settings=solver_settings(rebalancing.solver),
        re_optimize_criteria={
            "type": rebalancing.trigger,
            "threshold": rebalancing.pct_change_threshold,
        },
        api_settings=ApiSettings(scale_risk_aversion=False),
        risk_free_rate=risk_free_rates,
        decision_observer=recorder,
    )
    results, re_optimize_dates, value = rebalancer.re_optimize(
        transaction_cost_factor=rebalancing.transaction_cost
    )
    seconds = time.perf_counter() - start
    readings = backtest_readings(
        results,
        value,
        rebalancer.buy_and_hold_cumulative_portfolio_value,
        rebalancing.trading_days // rebalancing.look_forward_window,
        risk_free_rates,
    )
    decided = [decision.date for decision in recorder.decisions]
    if decided != [dates[window.start], *re_optimize_dates]:
        raise ValueError(
            f"The run recorded decisions on {decided}, but optimized on the "
            f"window's first day and on {re_optimize_dates}."
        )
    realized = utils.calculate_returns(
        load_prices(universe_path).iloc[window.start : window.end + 1],
        returns_compute_settings=RETURNS_SETTINGS,
    )["returns"]
    return {
        "trial": trial.index,
        "arm": arm.name,
        "fit_type": arm.fit_type,
        "device": arm.device,
        "assets_hash": trial.assets_hash,
        "trading_start": dates[window.start],
        "trading_end": dates[window.end],
        "risk_aversion": float(params.risk_aversion),
        "seconds": seconds,
        **readings,
        **chosen_readings(recorder.decisions, realized),
    }


def rebalancing_identity(
    config: ExperimentConfig,
    arms: Sequence[AnyArm],
    prices: pd.DataFrame,
    services: dict[str, dict],
) -> dict:
    """Everything a rebalancing results file depends on, apart from the trial count."""
    identity = run_identity(config, arms, prices, (), services)
    del identity["measurements"]
    extra = ("cvxpy", "cuopt") if config.rebalancing.solver == "CUOPT" else ("cvxpy",)
    identity["packages"] = package_versions(arms, extra)
    identity["host"] = host_metadata(arms)
    identity["rebalancing"] = config.rebalancing.model_dump(
        mode="json", exclude={"output", "scenario_results", "arms", "trials"}
    )
    identity["risk_free_rates"] = {
        "file": config.rebalancing.risk_free_rates.name,
        "sha256": hashlib.sha256(
            config.rebalancing.risk_free_rates.read_bytes()
        ).hexdigest(),
    }
    return identity


def run_rebalancing_experiment(
    config: ExperimentConfig, command: Sequence[str] = ()
) -> pd.DataFrame:
    """Run every pending (trial, arm) backtest and append it to the results file.

    Rows already in ``config.rebalancing.output`` are skipped, so a run resumes
    and a larger ``rebalancing.trials`` extends it; a file measured under other conditions
    is refused, as for the scenario comparison.

    Raises
    ------
    ValueError
        If the existing file was measured under another identity, a JointFM
        service serves another checkpoint, or a run does not cover its steps.
    """
    rebalancing = config.rebalancing
    prices = load_prices(config.prices)
    rates = risk_free.load_risk_free_rates(rebalancing.risk_free_rates)
    arms, services = resolve_arms(rebalancing.arms)
    identity = rebalancing_identity(config, arms, prices, services)
    frame = None
    commands = [list(command)]
    if rebalancing.output.exists():
        frame, stored = read_results(rebalancing.output)
        identity = merge_identity(rebalancing.output, stored.identity, identity)
        commands = stored.commands + commands
    check_same_checkpoint(identity["services"])
    metadata = RunMetadata(identity=identity, commands=commands)
    done = (
        set()
        if frame is None
        else set(frame[list(KEY_COLUMNS)].itertuples(index=False, name=None))
    )
    work = [
        (index, arm)
        for index in range(rebalancing.trials)
        for arm in arms
        if (index, arm.name) not in done
    ]
    trials: dict[int, tuple] = {}
    added = 0
    total = rebalancing.trials * len(arms)
    progress = tqdm(
        work,
        desc="rebalancing",
        unit="run",
        total=total,
        initial=total - len(work),
        mininterval=PROGRESS_INTERVAL_TERMINAL
        if sys.stderr.isatty()
        else PROGRESS_INTERVAL_LOG,
    )
    for index, arm in progress:
        if index not in trials:
            trial = draw_trial(
                prices,
                index,
                seed=config.seed,
                n_assets=config.assets,
                n_origins=config.origins,
                origin_history=config.origin_history,
                forward=config.forward,
            )
            window = draw_trading_window(
                prices,
                index,
                seed=config.seed,
                trading_days=rebalancing.trading_days,
                look_back_days=rebalancing.look_back_days,
            )
            universe_path = write_universe(prices, trial, rebalancing.universe_dir)
            params = calibrated_cvar_params(
                prices[list(trial.assets)], window, rebalancing
            )
            trials = {index: (trial, window, universe_path, params)}
        for called in jointfm_calls(arm):
            verify_jointfm_service(called, services[called.name])
        row = run_backtest(arm, *trials[index], config, rates)
        for called in jointfm_calls(arm):
            verify_jointfm_service(called, services[called.name])
        new = pd.DataFrame([row])
        frame = new if frame is None else pd.concat([frame, new], ignore_index=True)
        write_results(rebalancing.output, frame, metadata)
        added += 1
        progress.set_postfix_str(
            f"trial {index} {arm.name} step {row['mean_step_return']:+.4f} "
            f"{row['seconds']:.0f}s",
            refresh=False,
        )
    print(
        f"Rebalancing finished: {added} new runs, "
        f"{0 if frame is None else len(frame)} in {rebalancing.output}.",
        flush=True,
    )
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.simulator_comparison.rebalancing",
        description="Rebalancing backtest of every trial's universe per simulator, "
        "followed by its report.",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="YAML configuration file."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    # Before cvxpy, cuOpt, or cuML is imported, so they see the GPU .env selects.
    apply_gpu_selection()
    run_rebalancing_experiment(
        config, command=sys.argv if argv is None else ["main", *argv]
    )
    from .rebalancing_report import write_rebalancing_report

    outputs = write_rebalancing_report(
        config.rebalancing.output,
        config.rebalancing.scenario_results,
        config.rebalancing.report_dir,
    )
    print(
        f"Rebalancing report finished: {len(outputs)} files in "
        f"{config.rebalancing.report_dir}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

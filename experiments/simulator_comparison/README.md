<!-- SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Simulator Comparison

Runs the scenario simulators on identical inputs and records, per trial, forecast origin, and arm, how long `generate_cvar_data` took and how well the scenarios it returned forecast the realized next-day returns (`scores.py`).

- **Arm**: one simulator on one device: `no_fit-cpu` (the look-back window's own returns), `gaussian-cpu`, `kde-cpu` (scikit-learn), `kde-gpu` (cuML), `jointfm-cpu`, and `jointfm-gpu`. The config names the JointFM arm `jointfm`. It runs as `jointfm-cpu` or `jointfm-gpu`, depending on the device the service selected in `.env` reports.
- **Trial**: a seeded draw of `assets` price columns and `origins` forecast origins. Every arm gets the same assets, look-back window, and scenario seed at each origin, so arms differ only in the simulator. An origin needs `look_back` returns before it and `forward` returns after it. Only the scores read the forward returns; no arm sees them.

## Run

A YAML file in [`configs/`](configs/) describes a run completely: data, trial and origin counts, universe size, windows, scenario count, seed, and arms. Services and GPUs are not part of it: the arms read `.env` through the library's defaults, as every user of the library does. [`config.py`](config.py) validates it and describes every field. Every field must be present, and unknown keys are refused. The same file drives the report, which reads the results from its `output`. Copy a shipped config to change an experiment, for example to run only the CPU arms on a host without a GPU.

| Config | Purpose |
| --- | --- |
| [`pilot.yaml`](configs/pilot.yaml) | A short run to measure cost per arm and noise before sizing the full run |
| [`full.yaml`](configs/full.yaml) | The full comparison |
| [`timing.yaml`](configs/timing.yaml) | JointFM on CPU against GPU, sized for the CPU service (`task experiments:timing`) |

The JointFM arm and the cuML arm read `.env` in the repository root, the file the library's README describes: its JointFM selector picks the service, and `CUDA_VISIBLE_DEVICES` picks the GPU. The JointFM arm takes its expected returns from its own scenarios, the library's default (`JointFMSettings.expected_return="scenario"`); the other arms use the look-back window's mean, which is also what their scenarios are centered on. To compare JointFM on both devices, start a service on each, serving the same checkpoint. Run the config with `.env` pointing at one, then point `.env` at the other and run the same config again. The second run adds the other device's JointFM arm to the same results file and skips every row already in it. The runner refuses JointFM arms whose services serve different checkpoints. Before every JointFM call it checks that `.env` still selects the service recorded at session start, so a selector switched during a session stops the run instead of recording the other device's calls under this arm.

```bash
task experiments:all                         # full.yaml: run and report everything, resuming its results
task experiments:fresh                       # full.yaml: delete its outputs, run and report everything
task experiments:pilot                       # configs/pilot.yaml: run, then report
task experiments:run CONFIG=experiments/simulator_comparison/configs/full.yaml
task experiments:timing GPU_URL=http://127.0.0.1:8080 CPU_URL=http://127.0.0.1:8081
```

`experiments:fresh` deletes every output the config names (scenario results and report, rebalancing results, report, and universes; [`clean.py`](clean.py) prints each path) and then runs `experiments:all`: the scenario comparison, its report, and the rebalancing backtests with their report. Use it after a change of code or library defaults, which the resumable runs would otherwise refuse or mix into old results. To continue an interrupted `experiments:fresh`, run `experiments:all`, which resumes both results files; rerunning `experiments:fresh` would delete what was already measured.

`experiments:timing` measures JointFM on both devices from scratch with [`timing.yaml`](configs/timing.yaml): one session against `GPU_URL`, one against `CPU_URL`, each set as `JOINTFM_LOCAL_BASE_URL` over `.env`, then the report and its CPU-to-GPU speedup. `.env` must use the local selector or none, because selectors from the process and `.env` count together.

Without Task, the same run is `uv run --extra cuda13 --extra jointfm --extra experiments python -m experiments.simulator_comparison --config <config>`.

A progress bar on stderr counts forecast origins against the whole run, so a resumed run starts at the origins already in the results file, and shows each arm's latest generation time. The rebalancing bar counts runs the same way. When stderr is not a terminal, it refreshes once a minute so a log file stays short.

Run the arms on an otherwise idle host. The arms at one origin run one after another, but other load on the CPU or the GPUs still shifts their timings.

## Results

Results go to the config's `output`; the shipped configs write to the git-ignored `results/` directory. The file has one row per (trial, origin, arm). Its Parquet schema metadata holds the run's identity: the price file's hash, the experiment parameters, each arm's settings, the measurement columns, the host's CPU and GPU, package versions, and every JointFM service's checkpoint and device. It also holds the command line of each session that added rows.

- **Resuming**: rerunning the same config skips rows already in the file, so an interrupted run resumes, and a larger `trials` extends a finished run. Any other change to the identity is refused instead of mixing conditions in one file.
- **Timing**: `seconds` is the wall time of one `generate_cvar_data` call. For JointFM it includes the client's health request and the HTTP round trip.
- **Warm-up**: the first call of each arm in a process pays for imports, CUDA and cuML start-up, or the first connection, and is flagged `warmup`. Leave flagged rows out of timing summaries.
- **Scores**: the columns of `ScenarioScores` in `scores.py`. CRPS, energy score, spread, 50 % and 90 % interval coverage, a 20-bin PIT histogram, correlation errors against the look-back window and the forward returns, and the equal-weight portfolio's 95 % VaR exceedance. Its docstring defines each column. Skill against `no_fit` needs that arm's row at the same origin, so the report computes it from the table.

## Report

```bash
task experiments:report CONFIG=experiments/simulator_comparison/configs/full.yaml
```

It reads only the config's results file, so it can be rerun at any time, also while a run is still adding trials. It uses only trials in which every arm has every origin, and prints how many it left out. While a session for one arm is still running, such as the CPU JointFM session, leave that arm out with `EXCLUDE=jointfm-cpu` (`--exclude` for the script) to report the finished arms over all their trials; rerun without it when the session ends. It writes its tables and plots to `<output stem>_report/` beside the results file.

- **Ranking** (`ranking.png`, `ranking.csv`): which arm gives the most realistic scenarios. Per trial, arms are ranked by energy score (lower is better). The report gives each arm's share of trials won and its mean rank.
- **Skill** (`skill.png`, `skill.csv`): per trial, the percent reduction of energy score and CRPS against `no_fit-cpu`, with the median and its 95 % bootstrap interval over trials.
- **Scores** (`scores.png`, `summary.csv`): every score per arm, aggregated per trial over its origins, as a point with its 95 % bootstrap interval. A score where lower is better shows each arm's median paired difference from `no_fit-cpu`, which removes the spread between trials that every arm shares; a score with a target (spread, coverage, correlation bias, VaR exceedance) shows each arm's mean over trials against the target.
- **Comparisons** (`comparisons.csv`): every arm against `no_fit-cpu`, and each simulator's GPU arm against its CPU arm, with the median paired difference, its bootstrap interval, and a Wilcoxon signed-rank p-value.
- **Calibration** (`calibration_pit.png`, `calibration_reliability.png`, `reliability.csv`): mean PIT histograms, and the empirical against nominal coverage of central intervals.
- **Timing** (`timing.png`, `timing.csv`, `speedup.csv`): seconds per generation call without warm-up, and each simulator's CPU-to-GPU time ratio.

Colors follow the simulator. Filled marks are CPU arms and hollow marks GPU arms, and every axis label names its arm.

## Rebalancing

```bash
task experiments:rebalance CONFIG=experiments/simulator_comparison/configs/full.yaml
```

This runs each trial's universe through the library's rebalancing backtest (`rebalance_portfolio`), once per simulator in the config's `rebalancing.arms`, and then writes its report to `<rebalancing output stem>_report/`.

- **Trials:** `rebalancing.trials` sets the count independently of the scenario comparison. Trial `i` trades the assets of scenario trial `i`, drawn from the experiment seed and `i` alone, so the trials both run share their universes.
- **Trading window:** each trial draws its own window of `trading_days` from its seed.
- **Rebalancing rule:** after every `look_forward_window` trading days, `trigger` decides whether the portfolio is re-optimized, with the configured transaction cost and CVaR constraints. `every_step` re-optimizes every time, a fixed calendar that gives every arm the same decision dates, so arms differ only in their scenarios. `pct_change` re-optimizes only when the step's return, or the sum of the losing steps since the last re-optimization, falls below `pct_change_threshold`.
- **Risk aversion:** the configured base is scaled once per trial on the history before the window, as the rebalancing notebook does. The runs keep that value, with no per-window scaling.
- **Cash:** cash earns the daily 3-month Treasury bill rate from `rebalancing.risk_free_rates`, which `task data:risk-free-rates` downloads from FRED (series DTB3). Every optimization credits cash with the rate quoted on its date, the backtest accrues the rate day by day, and the Sharpe and Sortino ratios subtract it. A day earns the rate quoted before it, compounded over 252 trading days a year.
- **Pairing:** all arms of a trial share the window, constraints, risk aversion and scenario seed.
- **Arms:** the device changes only speed, not the portfolios, so one arm per simulator is enough. `jointfm` reaches the service `.env` selects.

Each run records:

| Group | Readings |
|---|---|
| Per step | Mean step return (average PnL per step) and step-return volatility |
| Over the window | Total simple return, annualized return and volatility, maximum drawdown |
| Risk-adjusted | Sharpe and Sortino ratios against the daily risk-free rate, Calmar ratio |
| Trading | Turnover and rebalance count |
| Baseline | Buy-and-hold return of the first portfolio |
| Path | The per-step returns |

The report compares every reading of every arm with `no_fit-cpu`, paired by trial. It plots:
- every reading's median paired difference from `no_fit-cpu` with its 95 % bootstrap interval (`rebalancing_differences`);
- the mean cumulative return after each step (`rebalancing_paths`).

Rerunning resumes the results file and re-renders the report; `task experiments:rebalance-report` re-renders it without running anything. Portfolio values follow the library's backtester, which values `LOG` returns as `exp(cumsum(R_log @ w))`. That value can't reach zero, so it understates the losses of short positions.

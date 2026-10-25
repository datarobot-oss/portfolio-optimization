<!-- SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Simulator Comparison

Runs the scenario simulators on identical inputs and records, per trial, forecast origin, arm, and horizon, how long the arm took to generate its scenarios and how well they forecast the realized returns over the horizon of the assets and of random portfolios of them (`scores.py`).

- **Arm**: one simulator on one device: `no_fit-cpu` (the look-back window's own returns), `gaussian-cpu`, `kde-cpu` (scikit-learn), `kde-gpu` (cuML), `jointfm-cpu`, and `jointfm-gpu`. The config names the JointFM arm `jointfm`. It runs as `jointfm-cpu` or `jointfm-gpu`, depending on the device the service selected in `.env` reports. The three dependence swap arms, `swap-jointfm-window`, `swap-window-jointfm`, and `swap-window-window`, recombine two of these simulators instead of simulating ([Dependence swap arms](#dependence-swap-arms)). Two more arms attribute the rebalancing readings: `jointfm-historical`, JointFM's scenarios on the window's mean, and the positive control `window-independent`, the window's returns with all dependence removed ([Rebalancing attribution](#rebalancing-attribution)).
- **Horizon**: the number of trading days after the origin over which a scenario and its outcome run, one of the config's `horizons`. A rebalanced portfolio is held for weeks, so day-one scores cannot see what builds up over them: volatility clustering, mean reversion, and dependence that changes in a sell-off. [Horizons](#horizons) describes how each arm forecasts them.
- **Portfolio size**: the number of assets a scored portfolio holds, a reporting dimension of its own. An equal-volatility portfolio's variance scales with (1 − ρ)/N + ρ for N assets of mean pairwise correlation ρ, so size 1 reads the per-asset distributions alone and larger sizes read the dependence between assets ever more.
- **Trial**: a seeded draw of `assets` price columns and `origins` forecast origins. Every arm gets the same assets, look-back window, and scenario seed at each origin, so arms differ only in the simulator. An origin needs `origin_history` returns before it, at least the look-back window's `look_back`, and `forward` returns after it. Only the draw reads `origin_history`, so two configs that differ only in `look_back` draw the same trials, origins, and scenario seeds ([History comparison](#history-comparison)). Only the scores read the forward returns; no arm sees them.

## Run

A YAML file in [`configs/`](configs/) describes a run completely: data, trial and origin counts, universe size, windows, scenario count, seed, arms, portfolio sizes, and horizons. Services and GPUs are not part of it: the arms read `.env` through the library's defaults, as every user of the library does. [`config.py`](config.py) validates it and describes every field. Every field must be present, and unknown keys are refused. The same file drives the report, which reads the results from its `output`. Copy a shipped config to change an experiment, for example to run only the CPU arms on a host without a GPU.

| Config | Purpose |
| --- | --- |
| [`pilot.yaml`](configs/pilot.yaml) | A short run to measure cost per arm and noise before sizing the full run |
| [`full.yaml`](configs/full.yaml) | The full comparison |
| [`full_long_history.yaml`](configs/full_long_history.yaml) | `full.yaml` with a longer look-back window for every arm of the scenario comparison, at the same origins ([History comparison](#history-comparison)); its rebalancing settings are `full.yaml`'s |
| [`timing.yaml`](configs/timing.yaml) | JointFM on CPU against GPU, sized for the CPU service and scored at the next day only, so each timed call is one next-day request (`task experiments:timing`) |

The JointFM arm and the cuML arm read `.env` in the repository root, the file the library's README describes: its JointFM selector picks the service, and `CUDA_VISIBLE_DEVICES` picks the GPU. The JointFM arm takes its expected returns from its own scenarios, the library's default (`JointFMSettings.expected_return="scenario"`); the other arms use the look-back window's mean, which is also what their scenarios are centered on. To compare JointFM on both devices, start a service on each, serving the same checkpoint. Run the config with `.env` pointing at one, then point `.env` at the other and run the same config again. The second run adds the other device's JointFM arm to the same results file and skips every row already in it. The runner refuses JointFM arms whose services serve different checkpoints. Before every JointFM call it checks that `.env` still selects the service recorded at session start, so a selector switched during a session stops the run instead of recording the other device's calls under this arm.

```bash
task experiments:all                         # full.yaml: run and report everything, resuming its results
task experiments:fresh                       # full.yaml: delete its outputs, run and report everything
task experiments:pilot                       # configs/pilot.yaml: run and report everything, resuming its results
task experiments:pilot-fresh                 # configs/pilot.yaml: delete its outputs, run and report everything
task experiments:run CONFIG=experiments/simulator_comparison/configs/full.yaml
task experiments:timing GPU_URL=http://127.0.0.1:8080 CPU_URL=http://127.0.0.1:8081
```

`experiments:fresh` (and `experiments:pilot-fresh` for `pilot.yaml`) deletes every output the config names (scenario results and report, rebalancing results, report, and universes; [`clean.py`](clean.py) prints each path) and then runs `experiments:all`: the scenario comparison, its report, and the rebalancing backtests with their report. Use it after a change of code or library defaults, which the resumable runs would otherwise refuse or mix into old results. To continue an interrupted `experiments:fresh`, run `experiments:all`, which resumes both results files; rerunning `experiments:fresh` would delete what was already measured.

`experiments:timing` measures JointFM on both devices from scratch with [`timing.yaml`](configs/timing.yaml): one session against `GPU_URL`, one against `CPU_URL`, each set as `JOINTFM_LOCAL_BASE_URL` over `.env`, then the report and its CPU-to-GPU speedup. `.env` must use the local selector or none, because selectors from the process and `.env` count together.

Without Task, the same run is `uv run --extra cuda13 --extra jointfm --extra experiments python -m experiments.simulator_comparison --config <config>`.

A progress bar on stderr counts forecast origins against the whole run, so a resumed run starts at the origins already in the results file, and shows each arm's latest generation time. The rebalancing bar counts runs the same way. When stderr is not a terminal, it refreshes once a minute so a log file stays short.

Run the arms on an otherwise idle host. The arms at one origin run one after another, but other load on the CPU or the GPUs still shifts their timings.

## Results

Results go to the config's `output`; the shipped configs write to the git-ignored `results/` directory. The file has one row per (trial, origin, arm, horizon). Its Parquet schema metadata holds the run's identity: the price file's hash, the experiment parameters, each arm's settings, the measurement columns, the host's CPU and GPU, package versions, and every JointFM service's checkpoint and device. It also holds the command line of each session that added rows.

- **Resuming**: rerunning the same config skips rows already in the file, so an interrupted run resumes, and a larger `trials` extends a finished run. Any other change to the identity, the horizons included, is refused instead of mixing conditions in one file.
- **Timing**: `seconds` is the wall time of the one call that generated an arm's scenarios at every horizon of an origin, repeated in each of its rows. With `horizons: [1]` the call is one `generate_cvar_data` call; with longer horizons it is one request for JointFM, and for every other arm its next-day call plus one one-day draw per day up to the longest horizon. For JointFM it includes the client's health request and the HTTP round trip.
- **Warm-up**: the first call of each arm in a process pays for imports, CUDA and cuML start-up, or the first connection, and is flagged `warmup`. Leave flagged rows out of timing summaries.
- **Scores**: the columns of `ScenarioScores` in `scores.py`, against the realized log return over the row's horizon of h days, in units of each asset's trailing daily volatility times √h. CRPS, energy score, variogram score of order 0.5 ([Scheuerer & Hamill 2015](https://doi.org/10.1175/MWR-D-14-00269.1), which compares every asset pair's expected absolute return difference with the realized one and so reacts to wrong correlations that the energy score barely sees), spread, 50 % and 90 % interval coverage, a 20-bin PIT histogram, correlation errors against the daily returns of the look-back window and of the forward window, and the equal-weight portfolio's 95 % VaR exceedance. One column scores the arm's expected returns instead of its scenarios: `mean_correlation`, the correlation across assets of the next-day `CvarData.mean` its rebalancing runs optimize on with the realized h-day log returns, 0 for expected returns that carry no information. Every arm on the window's mean reads the same value; JointFM's scenario mean is read off the next day of its multi-horizon request. Results files recorded before this column cannot be resumed or reported. Its docstring defines each column. Skill against `no_fit` needs that arm's row at the same origin, so the report computes it from the table.
- **Portfolio scores**: the columns of `PortfolioScores` in `scores.py`. At every origin, [`portfolios.py`](portfolios.py) draws five asset subsets per size of `portfolio_sizes` from the origin's scenario seed, so every arm and every horizon scores the same portfolios. Over a horizon of h days a portfolio holds its weights from the origin on, without rebalancing. Each subset carries an equal-weight portfolio, fully invested, and, where `rebalancing.cvar`'s weight, cash, and leverage bounds allow one at that size, a random long-only and a random long-short portfolio. Under the shipped bounds a long-short portfolio needs two long positions besides its shorts, so it starts at size 3. Per (kind, size) cell, the mean over its subsets of: the 95 % and 99 % VaR breaches, the CRPS in units of the portfolio's trailing daily volatility times √h, the squared standardized error, and a 20-bin PIT histogram; `portfolio_crps` averages the CRPS over every cell. At the full universe, the equal-weight cell's 95 % breach equals `ew_var_exceeded`. Results files recorded before these columns or the horizons existed cannot be resumed or reported; rerun them from scratch.

## Horizons

At every origin, each arm generates its scenarios for every horizon of `horizons` in one call ([`horizons.py`](horizons.py)), and every horizon is scored on its own.

- **JointFM** forecasts every horizon in one request and reads the h-day log return as the forecast log price h days after the origin minus the last observed one (`generate_horizon_samples_jointfm`). The service draws each horizon independently, so scenario `s` at one horizon and scenario `s` at another are not one path; the scores never pair them.
- **Every other arm** builds its h-day scenario as the sum of h independent one-day draws of its own simulator: the one rule that turns each baseline into an h-day forecast without new modeling, and for `gaussian` the familiar √h scaling. Each day's draw comes from the arm's simulator under its own seed, derived from the origin's scenario seed. `no_fit`'s one-day scenarios are the window's fixed returns, so its daily draws resample them uniformly with replacement, `num_scen` per day. The sums are read off one running total up to the longest horizon, so the longest horizon sets the number of draws.
- **The next day** is every arm's own one-day scenarios, exactly as a run with `horizons: [1]` produces them. For the simulators the comparison runs itself, the next-day rows therefore do not depend on the other horizons. JointFM's next day comes out of the multi-horizon request instead, so its draws differ from those of a next-day-only request; comparing its next-day readings with those of a `horizons: [1]` run of the same config shows whether querying more horizons changes JointFM's next-step forecast.
- **Blind spots.** The correlation references stay the daily correlations of the look-back and forward windows at every horizon, which equal the h-day correlations only when days are independent; 63 forward days cannot estimate an h-day correlation. Two origins of a trial less than h days apart have overlapping h-day outcomes, so at long horizons the breach rates' binomial bands, which count each origin once, are narrower than the data support.

## Dependence swap arms

The swap arms tell whether a difference between JointFM and the window comes from JointFM's per-asset distributions (its marginals) or from how its assets move together (its dependence). They cross the two marginal sources with the two dependence sources in a two-by-two design. Each arm is named `swap-<marginals>-<dependence>`, and `window` is `no_fit-cpu`:

| Marginals \ Dependence | Window | JointFM |
| --- | --- | --- |
| **Window** | `swap-window-window` | `swap-window-jointfm` |
| **JointFM** | `swap-jointfm-window` | `jointfm` |

- **Construction** ([`swaps.py`](swaps.py)): per asset, the scenario that holds the dependence donor's k-th smallest value gets the marginal donor's k-th smallest value. This is the Schaake shuffle ([Clark et al. 2004](https://doi.org/10.1175/1525-7541(2004)005%3C0243:TSSAMF%3E2.0.CO;2)). Every asset keeps its marginal donor's distribution exactly, and the joint rank structure (the empirical copula) is the dependence donor's. The JointFM corner is `jointfm` itself, since its own rank order leaves it unchanged.
- **The window's rows:** at the next day the window donor holds the look-back window's `look_back` rows, so both donors are first stretched to `num_scen` rows by repeating each row equally often. A run with swap arms therefore needs `num_scen` of at least `look_back`, and of at least `rebalancing.look_back_days` when the rebalancing comparison runs them. A stretched rank template repeats ranks. The ties are broken at random, independently per asset, from the origin's scenario seed.
- **The window corner** `swap-window-window` is the window's rows, stretched. It goes through the same construction as the two swaps, so it is the corner they are compared with, and its difference from `no_fit-cpu` measures what the construction itself adds.
- **Donors run next to the cells.** A config lists every swap arm's donors as arms too, and refuses the arm otherwise. The cells read their donors' scenarios at each origin and horizon, so they add no JointFM request. A cell added to a results file whose donors already ran generates its donors again from the origin's seed, without rows. A cell's time is its shuffle alone.
- **In the rebalancing comparison** each decision of a cell draws both donors' scenarios with the seed their own arms use, through the library's `rebalance_portfolio(scenario_generator=...)`. The cell then optimizes on the shuffled scenarios with the marginal donor's expected returns, so it differs from its marginal donor only in dependence. A cell with a JointFM donor sends one more JointFM request per decision.

## Rebalancing attribution

JointFM's rebalancing arm differs from the window's in two ways at once: its scenarios and its expected returns, which the library takes from the scenarios' mean. Two arms tell the two apart and show whether the backtest sees dependence at all.

- **`jointfm-historical`** optimizes on JointFM's scenarios with the window's mean as expected returns (`JointFMSettings.expected_return="historical"`). It sends JointFM's requests with JointFM's seed, so it draws exactly JointFM's scenarios. `jointfm` against it isolates the expected returns, and it against `no_fit-cpu` isolates the scenarios. It runs only in `rebalancing.arms`: in the scenario comparison its scenarios would be JointFM's and its expected returns the window's, which `no_fit-cpu` already reads. Its device is the JointFM service's, and it needs `jointfm` listed too.
- **`window-independent`**, the positive control, is the window's scenarios with each asset's rows permuted independently ([`independence.py`](independence.py)). It keeps every asset's returns, and so their distribution and mean, and removes all dependence between assets. A reading it moves beyond its noise against `no_fit-cpu` can see dependence; a reading it does not move cannot judge a simulator's dependence. It runs in both comparisons, at every horizon on the window's scenarios at that horizon, and needs `no_fit-cpu` listed too. Its time is its permutation alone. The permutations come from the origin's scenario seed, or the rebalancing seed, and a stream of their own.

## Report

```bash
task experiments:report CONFIG=experiments/simulator_comparison/configs/full.yaml
```

It reads the config's results file and its price file, so it can be rerun at any time, also while a run is still adding trials. Every reading is computed per horizon: each table holds every horizon in its `horizon` column, the horizon figures draw every reading over the horizons, and the other figures show the next day. The price file must be the one the results were measured on: the report compares its SHA-256 with the one the results file recorded and refuses another. It uses only trials in which every arm has every origin at every horizon, and prints how many it left out. While a session for one arm is still running, such as the CPU JointFM session, leave that arm out with `EXCLUDE=jointfm-cpu` (`--exclude` for the script) to report the finished arms over all their trials; rerun without it when the session ends. It writes its tables and plots to `<output stem>_report/` beside the results file.

- **Ranking** (`ranking.png`, `ranking.csv`): which arm gives the most realistic scenarios, by three proper scores side by side: the energy score, the variogram score, and the portfolio CRPS. Per trial and score, arms are ranked (lower is better), and the report gives each arm's share of trials won and its mean rank. The rankings are never merged into one number, because any weights between them would be arbitrary; an arm that wins on the energy score but loses on the two dependence-aware scores stands out across the row.
- **Skill** (`skill.png`, `skill.csv`): per trial, the percent reduction of the three ranking scores and of CRPS against `no_fit-cpu`, with the median and its 95 % bootstrap interval over trials.
- **Portfolio sizes** (`portfolio_sizes.png`, `portfolio_sizes.csv`, `portfolio_pit_<kind>.png`, `portfolio_pit.csv`): every portfolio reading as a curve over the portfolio size, one panel per kind. The 95 % and 99 % breach rates come with the binomial band of a calibrated forecast over the arm's origins: inside the band is calibrated. The band counts each origin once, because the portfolios scored on one day share it. The spread ratio is `1 / sqrt(mean squared standardized error)` over all trials, against the target 1; below 1 the scenarios are too narrow. The CRPS skill is the median per-trial reduction of the cell's CRPS against `no_fit-cpu`. The PIT figures give each arm's mean histogram per size.
- **Scores** (`scores.png`, `summary.csv`): every score per arm, aggregated per trial over its origins, as a point with its 95 % bootstrap interval. A score where lower is better shows each arm's median paired difference from `no_fit-cpu`, which removes the spread between trials that every arm shares; a score with a target (spread, coverage, correlation bias, VaR exceedance) shows each arm's mean over trials against the target.
- **Comparisons** (`comparisons.csv`): every arm against `no_fit-cpu`, and each simulator's GPU arm against its CPU arm, with the median paired difference, its bootstrap interval, and a Wilcoxon signed-rank p-value.
- **Calibration** (`calibration_pit.png`, `calibration_reliability.png`, `reliability.csv`): mean PIT histograms, and the empirical against nominal coverage of central intervals.
- **Horizons** (`horizons.png`, `portfolio_horizons_<kind>.png`): every reading as a curve over the horizons, drawn per panel as in the other figures. `horizons.png` holds the scores and skills of `scores.png` and `skill.png`; each `portfolio_horizons_<kind>.png` holds the readings of `portfolio_sizes.png` for one portfolio kind, one column per portfolio size, with the breach rates' binomial bands. The numbers are those of the horizon rows of the other tables.
- **Timing** (`timing.png`, `timing.csv`, `speedup.csv`): seconds per generation call without warm-up, read from the next-day rows, and each simulator's CPU-to-GPU time ratio. A call generates every horizon, so a config's timing compares only with configs of the same horizons.
- **Regimes** (`regimes_<feature>.png`, `regimes.csv`, `origin_features.csv`, `regime_summary.csv`, `regime_comparisons.csv`, `regime_skill.csv`, `regime_ranking.csv`, `regime_portfolio_sizes.csv`): every score, skill, ranking, and portfolio-size reading above, repeated per tercile of three features of the forecast origin, because an average over all origins can hide that an arm gains in one market regime and fails in another. [`regimes.py`](regimes.py) recomputes the features from the returns each origin's arms and scores saw:
  - `window_correlation`: the look-back window's mean pairwise correlation.
  - `volatility_ratio`: each asset's volatility over the window's last 63 returns divided by its volatility over the whole window, median over assets. It is above 1 when volatility rose into the origin.
  - `correlation_change`: the forward returns' mean pairwise correlation minus the window's. It looks ahead, so it explains readings and never selects a forecast.

  Each table holds the overall readings first, with `feature` and `regime` set to `all` and the same values as the overall tables, then one block per feature and tercile (`low`, `middle`, `high`). Within a tercile, each trial is aggregated over its origins in that tercile, and the intervals resample trials. `regimes.csv` gives each tercile's feature bounds and its numbers of origins and trials, and `origin_features.csv` gives each origin's features and terciles. Each figure shows one feature: the first column of every panel holds all origins, the next three the terciles, drawn as in `scores.png` and `skill.png`. The portfolio-size readings per tercile are in `regime_portfolio_sizes.csv` only, since a curve per size, kind, and tercile would not fit one figure.

- **Swap effects** (`swap_effects.csv`, only when the results hold the three [dependence swap arms](#dependence-swap-arms)): every score and portfolio reading split per trial into the main effect of JointFM's marginals, ((JJ − WJ) + (JW − WW)) / 2, the main effect of its dependence, ((JJ − JW) + (WJ − WW)) / 2, and their interaction, (JJ − WJ) − (JW − WW). Cells are named with the marginal source first: J is JointFM and W is the window. A `construction` row gives the window corner minus `no_fit-cpu`, which should be zero within noise. Each row gives the median over trials, its 95 % bootstrap interval, and a Wilcoxon p-value.

Colors follow the simulator. Filled marks are CPU arms and hollow marks GPU arms, and every axis label names its arm.

## History comparison

```bash
task experiments:history BASELINE=experiments/simulator_comparison/configs/full.yaml CONFIG=experiments/simulator_comparison/configs/full_long_history.yaml
```

Reads what a longer history changes, from the results of two configs that differ only in `look_back` and their outputs. Both draw their origins from `origin_history`, so their rows pair trial by trial. Every arm gets the longer history, not only JointFM, because the correlation bias reads each arm against the window it saw: with JointFM alone on a longer history, its bias would mix what it keeps of the dependence with the difference between a long and a short window's correlations. [`history.py`](history.py) refuses two files whose identities differ in anything but `look_back`, the arms each holds, and the host, so both runs must read the same JointFM checkpoint. It writes `history_comparison.csv` into the `--config` report directory: per horizon, arm both files hold, and score, the median over trials of the long-history minus the baseline reading, with its 95 % bootstrap interval, Wilcoxon p-value, and number of paired trials. The regime readings of each run split its origins by features of its own window, so they do not pair across runs.

## Rebalancing

```bash
task experiments:rebalance CONFIG=experiments/simulator_comparison/configs/full.yaml
```

This runs each trial's universe through the library's rebalancing backtest (`rebalance_portfolio`), once per simulator in the config's `rebalancing.arms`, and then writes its report to `<rebalancing output stem>_report/`.

- **Trials:** `rebalancing.trials` sets the count independently of the scenario comparison. Trial `i` trades the assets of scenario trial `i`, drawn from the experiment seed and `i` alone, so the trials both run share their universes.
- **Trading window:** each trial draws its own window of `trading_days` from its seed.
- **Rebalancing rule:** after every `look_forward_window` trading days, `trigger` decides whether the portfolio is re-optimized, with the configured transaction cost and CVaR constraints. `every_step` re-optimizes every time, a fixed calendar that gives every arm the same decision dates, so arms differ only in their scenarios and expected returns. `pct_change` re-optimizes only when the step's return, or the sum of the losing steps since the last re-optimization, falls below `pct_change_threshold`.
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
| Chosen portfolios | Per decision: the chosen portfolio's one-day scenario VaR and CVaR at the optimizer's `confidence`, its scenario mean and standard deviation, its effective number of holdings, and, over the days it is held, its VaR breaches and squared standardized returns. Per run: the breach rate, the ratio of realized to scenario volatility, and the effective holdings averaged over the held days |

The report compares every reading of every arm with `no_fit-cpu`, paired by trial. It plots:
- every reading's median paired difference from `no_fit-cpu` with its 95 % bootstrap interval (`rebalancing_differences`);
- the mean cumulative return after each step (`rebalancing_paths`).

`rebalancing_chosen_risk.csv` reads the risk of the portfolios each arm chose ([`chosen.py`](chosen.py)). An optimizer seeks out the portfolios whose risk its scenarios understate most, which the random portfolios of the portfolio scores cannot show. A decision holds its portfolio from the day after it until the next decision, and the scenario and realized returns both follow the portfolio scores' convention: simple returns aggregated by weight, cash earning nothing. The effective number of holdings, `(sum |w|)^2 / sum w^2` over the asset weights, is `n` for `n` positions of equal size, so a chosen portfolio's readings can sit next to the portfolio-size curves. Per arm, the table gives the share of held days that breach the VaR, with its bootstrap interval over trials and the binomial band of a correct VaR, and the realized-to-scenario volatility ratio with its interval. Every arm understates the risk of the portfolio it picks, so `no_fit-cpu`'s row, not the nominal rate, is the reference. Results files recorded before the chosen-portfolio record existed cannot be resumed or reported; rerun them from scratch.

With the three dependence swap arms it also writes `rebalancing_swap_effects.csv`, which splits every reading as `swap_effects.csv` does.

With `jointfm-historical` it also writes `rebalancing_mean_attribution.csv` ([Rebalancing attribution](#rebalancing-attribution)), which splits every reading's difference between `jointfm` and `no_fit-cpu`, per trial, into an `expected_returns` part, `jointfm` minus `jointfm-historical`, and a `scenarios` part, `jointfm-historical` minus `no_fit-cpu`; a `total` row gives the difference itself. Each row gives the mean over trials, its 95 % bootstrap interval, and a Wilcoxon p-value. It reports means rather than the medians of the comparisons, because means add up: the two parts' means sum to the total's. Whether a reading can see dependence at all reads off `window-independent`'s row in `rebalancing_comparisons.csv`.

Rerunning resumes the results file and re-renders the report; `task experiments:rebalance-report` re-renders it without running anything. Portfolio values follow the library's backtester, which values `LOG` returns as `exp(cumsum(R_log @ w))`. That value can't reach zero, so it understates the losses of short positions.

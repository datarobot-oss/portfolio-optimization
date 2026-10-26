<!-- AI agents: Start by reading AGENTS.md -->
# Portfolio Optimization Powered by NVIDIA cuOpt

[![Build Status](https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization/actions/workflows/main.yml/badge.svg)](https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization/actions/workflows/main.yml)
[![Checks](https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization/actions/workflows/checks.yml/badge.svg)](https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization/actions/workflows/checks.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Contributing](https://img.shields.io/badge/Contributing-Guide-green)](CONTRIBUTING.md)

## Disclaimer
This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

---

## Overview

This portfolio optimization developer example addresses the financial industry's trade-off between **computational speed** and **model complexity**. By leveraging **NVIDIA accelerated computing** — **[NVIDIA cuOpt](https://github.com/NVIDIA/cuopt)** for GPU-accelerated portfolio solves, and **[RAPIDS cuML](https://github.com/rapidsai/cuml)** for GPU scenario generation — this solution transforms robust analysis (e.g., Mean-CVaR, large-scale simulations) from slow batch processing into a **fast, iterative workflow** for dynamic decision-making.

### Accelerated Architecture

The end-to-end pipeline connects market data ingestion to optimal strategy backtesting using the NVIDIA CUDA ecosystem:

#### 1. Data Science & Scenario Generation
* **Technology:** **CUDA-X Data Science** — **[RAPIDS cuML](https://github.com/rapidsai/cuml)** for GPU KDE scenario generation
* **Function:** Accelerates data preprocessing and the learning/sampling of return distributions.
* **Performance:** Achieves speedups of up to **100x** when generating scenarios.

#### 2. Mean-CVaR Optimization
* **Technology:** **[NVIDIA cuOpt](https://github.com/NVIDIA/cuopt)** open-source solvers.
* **Function:** Efficiently solves complex, scenario-based **Mean-CVaR portfolio optimization** problems.
* **Performance:** Consistently outperforms state-of-the-art CPU-based solvers, with up to **160x speedups** in large-scale problems.

#### 3. Strategy Backtesting & Refinement
* **Technology:** **CUDA-X Data Science** and **HPC SDK**.
* **Function:** Rigorously tests the **trading strategies** and provides insights into strategy fine-tuning.

### Key Takeaways

* **Speed-ups:** Up to **160x faster** optimization and **100x faster** scenario generation.
* **Risk Modeling:** Enables the use of **Conditional Value-at-Risk (CVaR)** at production speed.
* **Iterative Workflow:** Supports dynamic, fast, and data-driven optimization cycles.

<p align="center">
    <img src="./docs/arch_diagram.png" alt="architecture diagram for PO" width="750"/>
</p>

---
## Get Started
### System Requirements
<details>
<summary><b>Recommended Requirements for Best Performance</b></summary>

- **System Architecture**:
  - x86-64
  - ARM64
- **GPU**:
  - NVIDIA H100 SXM (compute capability >= 9.0) and above
- **CPU**:
  - 32+ cores
- **System Memory**:
  - 64+ GB RAM
- **NVMe SSD Storage**:
  - 100+ GB free space
- **CUDA**:
  - 13.0
- **NVIDIA Drivers**:
  - Latest NVIDIA drivers (580.65.06+)
- **OS**:
  - Linux distributions with glibc>=2.28 (released in August 2018):
    - Arch Linux (minimum version 2018-08-02)
    - Debian (minimum version 10.0)
    - Fedora (minimum version 29)
    - Linux Mint (minimum version 20)
    - Rocky Linux / Alma Linux / RHEL (minimum version 8)

The above configuration will provide optimal performance for large-scale optimization problems.

</details>

### Installation on PyTorch Container

To install dependencies on the NVIDIA PyTorch container:

```bash
# Start the container. Publish 8888 for Jupyter and 8501 for Streamlit.
docker run --gpus all -it --rm \
  -v ./:/workspace/host \
  --ipc=host \
  -p 8888:8888 \
  -p 8501:8501 \
  nvcr.io/nvidia/pytorch:25.10-py3

# Clone the repository
git clone https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization.git
cd portfolio-optimization

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# To add $HOME/.local/bin to your PATH, either restart your shell or run:
source $HOME/.local/bin/env  # (sh, bash, zsh)
# source $HOME/.local/bin/env.fish  # (fish)

# Install with CUDA-specific dependencies, plus the Jupyter kernel tooling
uv sync --extra cuda13 --group notebooks # full CUDA 13 stack currently tracks cuOpt/cuML 26.04
# On CUDA 12 hosts, use the full cuOpt/cuML 26.06 stack:
# uv sync --extra cuda12 --group notebooks
# For direct SOCP testing on CUDA 13 with cuOpt 26.06:
# uv sync --extra cuda13-socp --group notebooks
# For JointFM scenario generation (fit_type="jointfm"), add the jointfm-client SDK:
# uv sync --extra cuda13 --extra jointfm --group notebooks

# Optional: Install development tools (ruff, pytest, pre-commit)
uv sync --extra cuda13 --group notebooks --group dev

# Create a Jupyter kernel for this environment
uv run python -m ipykernel install --user --name=portfolio-opt --display-name "Portfolio Optimization"

# Launch Jupyter Lab
uv run jupyter lab --no-browser --NotebookApp.token=''
```

**Note:** If you use a different container image than the suggested one above, during uv sync, use `--extra cuda12` for the full cuOpt/cuML 26.06 CUDA 12 stack or `--extra cuda13` for the current full CUDA 13 cuOpt/cuML stack. As of the cuOpt 26.06 release, `cuml-cu13` 26.06 is not published, so CUDA 13 SOCP testing with cuOpt 26.06 uses `--extra cuda13-socp`; that extra is cuOpt-only and is intended for direct SOCP preview/Mean-Variance variance-cap solves, not GPU KDE/CVaR rebalancing. The `uv sync` command automatically creates a virtual environment and installs all dependencies from `uv.lock`.

**Tip:** To check your CUDA version, run `nvidia-smi` and look for "CUDA Version" in the output.

**Important Notes:**
- If you encounter "No space left on device" errors, set `UV_CACHE_DIR` to an alternate cache location: `export UV_CACHE_DIR=/path/to/cache/directory`
- The `cuda12`, `cuda13`, and `cuda13-socp` extras are mutually exclusive - install only one based on your system's CUDA version and workflow
- If you plan to run the Streamlit demo from this container, include `-p 8501:8501` when starting Docker. Docker port mappings cannot be added to an already-running container; restart the container with the port published if it was omitted.

#### Using the Jupyter Kernel

After launching Jupyter Lab:
1. Navigate to the [`notebooks/`](notebooks/) directory
2. Open any notebook (e.g., `cvar_basic.ipynb`)
3. Select the "Portfolio Optimization" kernel from the kernel selector in the top-right corner
4. If the kernel is not visible, refresh the page or restart Jupyter Lab

To list all available kernels:
```bash
jupyter kernelspec list
```

To remove the kernel later (if needed):
```bash
jupyter kernelspec uninstall portfolio-opt
```

### Quick Start Locally

Explore the example notebooks in the [`notebooks/`](notebooks/) directory:
- **`cvar_basic.ipynb`**: Complete walkthrough of Mean-CVaR portfolio optimization with GPU acceleration
- **`efficient_frontier.ipynb`**: A quick tutorial on how to generate efficient frontier.
- **`rebalancing_strategies.ipynb`** Introduction to dynamic re-balancing and examples of testing strategies
- **`cvar_jointfm.ipynb`**: Mean-CVaR optimization on scenarios from the JointFM foundation model (see [JointFM Scenarios](#jointfm-scenarios))

### GPU Selection

cuOpt, cuML, and CuPy use the GPUs that `CUDA_VISIBLE_DEVICES` lists, one index or a comma-separated list (`1` or `1,2`); work inside one process runs on the first of them. Without the variable, `apply_gpu_selection` selects GPU 0 (`CUDA_VISIBLE_DEVICES=0`), and a script started without it sees every GPU but still runs on GPU 0. Set it in a `.env` file in the repository root, created from the template:

```bash
cp .env.sample .env  # then set CUDA_VISIBLE_DEVICES in .env
```

The variable must be set before anything imports cvxpy, because cvxpy creates the CUDA context while it probes its GPU solvers on import; setting it later is silently ignored. A variable already set in the process environment overrides the one in `.env`.

- **Notebooks.** The first code cell of every GPU notebook calls `apply_gpu_selection("../.env")` from `portfolio_optimization.gpu`, which applies the variable and refuses to run once cvxpy, cuOpt, cuML, or RMM is imported. Restart the kernel to change the selection.
- **Scripts and the Streamlit demo.** Start them with `uv run --env-file .env`, for example `uv run --env-file .env python my_script.py`, or call `apply_gpu_selection(".env")` before the script's first cvxpy import.

**Parallel runs.** `run_on_gpus(job, job_kwargs)` from `portfolio_optimization.gpu` runs independent jobs, such as whole rebalancing runs (`rebalance.run_rebalancing`) or scenario-generator arms, in spawned worker processes, one per GPU that `CUDA_VISIBLE_DEVICES` lists. Each worker sees only its own GPU, and an idle worker takes the next waiting job. With one GPU listed, the jobs run in sequence in the calling process. `rebalancing_strategies.ipynb` runs its strategies this way. A job must be a module-level function of an importable module, its arguments and result must be picklable, and a script that calls it with several GPUs needs an `if __name__ == "__main__":` guard. Worker start-up (the cvxpy and cuOpt imports) takes seconds, so it pays off for whole runs, not for sub-second solves.

### JointFM Scenarios

`fit_type="jointfm"` draws scenarios from [JointFM](https://github.com/datarobot/joint-client-python), a foundation model that forecasts the joint distribution of several time series, instead of fitting a KDE or Gaussian to the look-back window. Each scenario is one joint draw of the next period's asset returns, forecast from the look-back window, so it stands for the same quantity as the other fit types and everything downstream of `generate_cvar_data` is unchanged.

The library reaches JointFM through the [`jointfm-client`](https://pypi.org/project/jointfm-client/) SDK, installed with the optional `jointfm` extra:

```bash
uv sync --extra jointfm --group notebooks  # add your CUDA extra as above
```

The SDK reads its connection settings from a `.env` file in the repository root. Copy the template and set exactly one selector in it; the same code then runs against any of the four, and only `.env` changes:

```bash
cp .env.sample .env
```

| Selector | JointFM service |
| --- | --- |
| `JOINTFM_LOCAL_BASE_URL` | An instance you run yourself |
| `JOINTFM_LOCAL_BASE_URLS` | Several instances serving the same checkpoint |
| `JOINTFM_DEPLOYMENT_ID` | One DataRobot deployment (also needs `DATAROBOT_ENDPOINT` and `DATAROBOT_API_TOKEN`) |
| `JOINTFM_DEPLOYMENT_IDS` | A pool of DataRobot deployments serving the same checkpoint (same credentials) |

`.env` is git-ignored, so the API token stays out of version control. A variable set in the process environment overrides the same variable in `.env`, but selectors from both places count together, so switching to another service means changing the selector in `.env`.

```python
from portfolio_optimization import cvar_utils
from portfolio_optimization.settings import JointFMSettings, ScenarioGenerationSettings

scenario_generation_settings = ScenarioGenerationSettings(
    num_scen=10000,
    fit_type="jointfm",
    jointfm_settings=JointFMSettings(dotenv_path=".env"),  # run from the repository root
)
returns_dict = cvar_utils.generate_cvar_data(returns_dict, scenario_generation_settings)
```

- **Universe size.** All assets go into one request, because a scenario is a draw from their joint distribution. The universe must fit the served model's `max_series`, which the service reports in its health response; a larger universe is refused rather than split.
- **Log prices.** The simulator rebuilds each asset's log-price path from the look-back window's returns: the cumulative sum of its log returns (`LINEAR` returns converted with `log1p`), starting from 0 at the price before the window. It forecasts the next log price, subtracts the path's last log price to get the next log return, and converts that into the configured return type (`expm1` for `LINEAR`). The path's starting constant does not matter, because the model reads each series relative to its last value. Any log return maps to a positive price, so prices implied by the scenarios are always positive. This needs `ReturnsComputeSettings(return_type="LOG" or "LINEAR", freq=1)`; other return types and overlapping returns are refused.
- **History.** A log-price path longer than the served model's `n_input` points is cut to its most recent `n_input` points.
- **Scenario count.** The SDK splits a large `num_scen` across requests and pooled instances, so it is not limited by one instance's capacity.
- **Expected return.** By default the optimizer's expected returns are the mean of the JointFM scenarios, so expected return and CVaR come from the same forecast. `JointFMSettings(expected_return="historical")` uses the look-back window's mean instead, as the other fit types do.
- **Longer horizons.** `cvar_utils.generate_horizon_samples_jointfm` forecasts the returns over several horizons, each a number of periods after the last observation up to the served model's `n_output`, in one request, from the same log-price path. The service draws each horizon independently, so the scenarios at two horizons are not paths.

See [`notebooks/cvar_jointfm.ipynb`](notebooks/cvar_jointfm.ipynb) for a complete example.

### Experiments

[`experiments/`](experiments/) holds experiments that measure the library's components; they are not part of the installed package. [`experiments/simulator_comparison/`](experiments/simulator_comparison/README.md) runs every scenario simulator on the same random asset universes and forecast dates. It scores how realistic each simulator's scenarios are against the realized returns over the next day and over longer horizons, and times scenario generation on CPU and on GPU.

A YAML file under [`experiments/simulator_comparison/configs/`](experiments/simulator_comparison/configs/) describes each experiment completely, validated by a pydantic model. The [`Taskfile.yml`](Taskfile.yml) at the repository root runs and evaluates experiments with [Task](https://taskfile.dev/installation/), so its commands only name the config. `task --list` shows every task.

```bash
task setup                                   # reconcile uv.lock, install everything the tasks use, the pre-commit hooks, and the Jupyter kernel
task experiments:pilot                       # every experiment at a small size, resuming its results
task experiments:full                        # every experiment at full size, resuming its results
task experiments:all                         # run and report everything of full.yaml alone, resuming its results
task experiments:fresh                       # delete full.yaml's outputs, run and report everything
task experiments:run CONFIG=experiments/simulator_comparison/configs/full.yaml
task experiments:report CONFIG=experiments/simulator_comparison/configs/full.yaml
task experiments:test                        # CPU tests, no GPU or JointFM service needed
task data:risk-free-rates                    # download the T-bill rates the rebalancing backtests need
```

| Task | What it does |
| --- | --- |
| `setup` | Runs `uv lock`, which changes `uv.lock` only where `pyproject.toml` changed and keeps every other pin, then installs the locked environment with the `CUDA_EXTRA`, `jointfm`, and `experiments` extras and the `dev` and `notebooks` groups. It reports whether `uv.lock` has uncommitted changes. |
| `lock:upgrade` | Moves every pin in `uv.lock` to the newest version `pyproject.toml` allows, installs it, and runs `experiments:test`. Run it deliberately, between experiments whose results are compared, and commit `uv.lock` once the tests and a pilot run pass. |
| `experiments:pilot` | Runs every experiment at a small size: `pilot.yaml` and its next-day, long-history, and risk-averse twins, with every report and the history comparison, to measure cost per arm and noise before the full run. Rerunning resumes every results file. |
| `experiments:pilot-fresh` | Deletes every output of `experiments:pilot`, then runs it from scratch |
| `experiments:full` | Runs every experiment at full size: `full.yaml` and its three twins, as `experiments:pilot` does for the pilot set. Rerunning resumes every results file. |
| `experiments:full-fresh` | Deletes every output of `experiments:full`, then runs it from scratch |
| `experiments:run` | Runs a config (default `full.yaml`). Rerunning resumes its results file, and a larger `trials` extends it. |
| `experiments:report` | Writes the tables and plots of a config's results. `EXCLUDE` leaves out arms, for example one whose session is still running (`EXCLUDE=jointfm-cpu`). |
| `experiments:rebalance` | Runs a config's rebalancing backtests (each trial's universe, once per simulator) and writes their report: returns, drawdown, Sharpe, Sortino, and Calmar ratios, turnover |
| `experiments:rebalance-report` | Writes the rebalancing report of a config's results without running anything |
| `experiments:all` | Runs a config's scenario comparison (default `full.yaml`), the rebalancing backtests, and both reports. Rerunning resumes both results files, so it continues an interrupted `experiments:fresh`. |
| `experiments:fresh` | Deletes every output of a config (default `full.yaml`), then runs `experiments:all` from scratch |
| `experiments:timing` | Times JointFM on CPU against GPU from scratch (`timing.yaml`), one session per service: `GPU_URL` and `CPU_URL` |
| `experiments:test` | Runs the experiments' CPU tests |
| `data:risk-free-rates` | Downloads the daily 3-month Treasury bill rate (FRED DTB3) to `data/rates/dtb3.csv`, the rate cash earns in the rebalancing backtests |

The experiments need the `experiments` extra (`pyarrow`, `pyyaml`, `scipy`, `tqdm`), which the tasks install. `CUDA_EXTRA` (default `cuda13`) selects the CUDA extra for the GPU arm. The experiments read `.env` like the library does: its JointFM selector picks the service of the `jointfm` arm, which runs as `jointfm-cpu` or `jointfm-gpu` depending on that service's device. `CUDA_VISIBLE_DEVICES` in `.env` picks the GPU. To compare JointFM on both devices, use `experiments:timing`, or run a config once per service, switching the selector in `.env` in between; the second run adds the other device's arm to the same results file. A selector switched while a session runs stops it.

### Streamlit GTC Demo

The Streamlit demo from the GTC branch is available under [`demo/`](demo/) as a dynamic rebalancing app.

If you are using the PyTorch Docker container above, make sure it was started with `-p 8501:8501`. Streamlit must bind to `0.0.0.0` inside the container so the published Docker port can receive browser traffic.

```bash
uv pip install -r demo/requirements.txt
uv run python -c "from portfolio_optimization.utils import download_data; download_data('data/stock_data', datasets=['sp500'])"
uv run streamlit run demo/rebalancing_streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

For a remote GPU host, also forward port `8501` from your laptop to the host running Docker:

```bash
ssh -L 8501:localhost:8501 <user>@<remote-host>
```

See [`demo/README_streamlit.md`](demo/README_streamlit.md) for focused deployment instructions.

### Deploy on Brev
Deploy using [Brev launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-360InRZzyHqDnJYQKIxaSggF8xI): start an instance on Brev.nvidia.com and follow the instructions in the notebooks.


---
## Contribution Guidelines

We welcome contributions to this project! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:
- Code of conduct
- How to submit issues and feature requests
- Pull request process
- Coding standards and best practices

---
## Community

For questions, discussions, and community support:
- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization/issues)
- **Discussions**: Join conversations in [GitHub Discussions](https://github.com/NVIDIA-AI-Blueprints/portfolio-optimization/discussions)

---
## References

- [NVIDIA cuOpt](https://github.com/NVIDIA/cuopt) — source repository
- [NVIDIA cuOpt Documentation](https://docs.nvidia.com/cuopt/)
- [RAPIDS cuML](https://github.com/rapidsai/cuml) — source repository
- [RAPIDS cuML Documentation](https://docs.rapids.ai/api/cuml/stable/)
- Markowitz, H. (1952). "Portfolio Selection". *The Journal of Finance*, 7(1), 77-91.
- Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of conditional value-at-risk". *Journal of Risk*, 2, 21-42.

---
## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

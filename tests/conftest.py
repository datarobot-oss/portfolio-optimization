# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the portfolio optimization test suite."""

from types import SimpleNamespace

import numpy as np
import pytest

from portfolio_optimization import cvar_utils

# cuOpt gained QCQP/SOCP support in the 26.06 line. Earlier builds raise
# "Quadratic constraints not supported" from QuadraticExpression.__le__, so
# variance-cap tests must skip rather than fail against them -- the `cuda13`
# extra still tracks cuOpt 26.04 (see pyproject `cuda13` vs `cuda13-socp`).
CUOPT_SOCP_MIN_VERSION = (26, 6)


@pytest.fixture()
def require_cuopt_socp():
    """Skip unless the installed cuOpt can express quadratic constraints."""
    cuopt = pytest.importorskip("cuopt", reason="cuOpt GPU runtime required")
    raw = str(getattr(cuopt, "__version__", "0.0"))
    try:
        version = tuple(int(part) for part in raw.split(".")[:2])
    except ValueError:  # pragma: no cover - unparseable version string
        pytest.skip(f"cannot parse cuOpt version {raw!r}")
    if version < CUOPT_SOCP_MIN_VERSION:
        pytest.skip(
            f"cuOpt {raw} lacks QCQP/SOCP support; "
            f"{'.'.join(map(str, CUOPT_SOCP_MIN_VERSION))}+ required"
        )
    return version


class FakeJointFMClient:
    """Stand-in for ``jointfm_client.JointFMClient`` that records its calls.

    It mirrors the parts of the SDK its callers read: the health metadata
    (capacity envelope, device, model and checkpoint versions) and a sample
    forecast whose ``samples`` axis order is ``(sample, horizon, column)`` and
    whose array is read-only. Draws scatter around each column's last history
    value, as future log prices do, one horizon per query time.
    """

    def __init__(self):
        self.envelope = SimpleNamespace(
            min_series=1, max_series=10, n_input=40, n_output=100
        )
        self.device = "cpu"
        self.checkpoint_version = "fake-checkpoint"
        self.sample_scale = 1.0
        self.from_env_kwargs = None
        self.forecast_calls = []

    def health(self, *, cache=False):
        return SimpleNamespace(
            data_generation=self.envelope,
            device=self.device,
            model_version="fake-model",
            checkpoint_version=self.checkpoint_version,
            generation="jointfm-1",
            image_version="fake-image",
            schema_version="v6",
        )

    def forecast(self, history, *, n_samples, requested_columns, **kwargs):
        rng = np.random.default_rng(kwargs["seed"])
        shape = (n_samples, len(kwargs["query_times"]), len(requested_columns))
        samples = history[requested_columns].iloc[-1].to_numpy() + (
            self.sample_scale * rng.normal(size=shape)
        )
        samples.flags.writeable = False
        self.forecast_calls.append(
            dict(
                history=history,
                n_samples=n_samples,
                requested_columns=requested_columns,
                samples=samples,
                **kwargs,
            )
        )
        return SimpleNamespace(
            samples=samples, requested_columns=tuple(requested_columns)
        )


def _install_fake_jointfm(patch: pytest.MonkeyPatch) -> FakeJointFMClient:
    client = FakeJointFMClient()

    def from_env(**kwargs):
        client.from_env_kwargs = kwargs
        return client

    module = SimpleNamespace(
        JointFMClient=SimpleNamespace(from_env=from_env), __version__="0.0.fake"
    )

    def import_fake_client():
        return module

    patch.setattr(cvar_utils, "_import_jointfm_client", import_fake_client)
    return client


@pytest.fixture()
def fake_jointfm(monkeypatch):
    return _install_fake_jointfm(monkeypatch)


@pytest.fixture(scope="module")
def module_fake_jointfm():
    """``fake_jointfm`` for a module whose fixtures run an experiment once.

    The client stays installed until the module's last test has run.
    """
    with pytest.MonkeyPatch.context() as patch:
        yield _install_fake_jointfm(patch)


@pytest.fixture(scope="module")
def device_sessions(module_fake_jointfm):
    """Run both sessions of a CPU-against-GPU experiment on a host without a GPU.

    The GPU pipeline's cuOpt solves with Clarabel instead, and each session's
    environment check (``devices.session_record``) returns a fixed record with
    a stand-in GPU, so the sessions differ only in the device the fake JointFM
    service reports. The returned function takes the experiment's
    ``DevicesConfig`` and ``ExperimentConfig`` and returns its results file.
    """
    # Imported here: the library's tests run without the experiments extra.
    import dataclasses
    import os

    from experiments.simulator_comparison import devices, solvers

    module_fake_jointfm.envelope.n_input = 1000
    module_fake_jointfm.sample_scale = 0.01

    def record(device, service):
        session = {"jointfm": service}
        if device == "GPU":
            session["gpu"] = {"name": "Test GPU", "driver": "0", "uuid": "GPU-test"}
        return os.getpid(), session

    def run(device_config, experiment):
        for device, service_device in (("GPU", "cuda"), ("CPU", "cpu")):
            module_fake_jointfm.device = service_device
            devices.run_device_session(device_config, experiment, device)
        module_fake_jointfm.device = "cpu"
        return device_config.output

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(
            solvers.SOLVERS_BY_NAME,
            "CUOPT",
            dataclasses.replace(
                solvers.SOLVERS_BY_NAME["CUOPT"],
                backend="CLARABEL",
                options=(),
                module="clarabel",
            ),
        )
        patch.setattr(devices, "session_record", record)
        yield run


@pytest.fixture()
def window_scenario_results():
    """Run a config's scenario comparison with the window's arm alone.

    The rebalancing report draws the chosen portfolios over the size curves of
    the scenario results its config names; ``no_fit-cpu`` needs no service, so
    it is the cheapest arm that writes them. The returned function takes the
    config and returns its ``rebalancing.scenario_results``.
    """
    # Imported here: the library's tests run without the experiments extra.
    from experiments.simulator_comparison.arms import WINDOW_ARM
    from experiments.simulator_comparison.runner import run_experiment
    from experiments.simulator_comparison.scores import (
        PortfolioScores,
        ScenarioScores,
    )

    def run(config):
        portfolios = PortfolioScores(
            config.portfolio_sizes, config.rebalancing.cvar, config.assets
        )
        window = config.model_copy(update={"arms": (WINDOW_ARM,)})
        run_experiment(window, [ScenarioScores(), portfolios])
        return config.rebalancing.scenario_results

    return run


@pytest.fixture()
def saved_figures(monkeypatch):
    """The texts of every figure the reports save, by figure name.

    Each report module's ``_save`` is replaced by a recorder that keeps the
    figure's texts (titles, tick labels, legend entries), so a test reads
    which arms a figure names; nothing is written, which keeps the reports
    fast, and the reports get back the paths they would have written.
    """
    # Imported here: the library's tests run without the experiments extra.
    from matplotlib.text import Text

    from experiments.simulator_comparison import (
        chosen_report,
        dilution,
        portfolio_report,
        rebalancing_report,
        report,
    )

    texts = {}

    def record(figure, directory, name):
        texts[name] = "\n".join(text.get_text() for text in figure.findobj(Text))
        return [directory / f"{name}.png", directory / f"{name}.svg"]

    for module in (
        report,
        dilution,
        portfolio_report,
        rebalancing_report,
        chosen_report,
    ):
        monkeypatch.setattr(module, "_save", record)
    return texts

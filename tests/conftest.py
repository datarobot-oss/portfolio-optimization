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


@pytest.fixture()
def fake_jointfm(monkeypatch):
    client = FakeJointFMClient()

    def from_env(**kwargs):
        client.from_env_kwargs = kwargs
        return client

    module = SimpleNamespace(
        JointFMClient=SimpleNamespace(from_env=from_env), __version__="0.0.fake"
    )

    def import_fake_client():
        return module

    monkeypatch.setattr(cvar_utils, "_import_jointfm_client", import_fake_client)
    return client

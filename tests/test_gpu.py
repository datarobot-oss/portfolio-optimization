# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GPU selection through CUDA_VISIBLE_DEVICES."""

import os
import subprocess
import sys
import types

import pytest

from portfolio_optimization.gpu import (
    CUDA_INITIALIZING_MODULES,
    CUDA_VISIBLE_DEVICES,
    apply_gpu_selection,
)


@pytest.fixture()
def fresh_process(monkeypatch):
    """A process state in which no CUDA library is imported and no GPU is selected."""
    # setenv first so monkeypatch restores the variable even when the test sets it.
    monkeypatch.setenv(CUDA_VISIBLE_DEVICES, "unused")
    monkeypatch.delenv(CUDA_VISIBLE_DEVICES)
    for name in CUDA_INITIALIZING_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    return monkeypatch


def write_dotenv(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text)
    return path


class TestApplyGpuSelection:
    def test_applies_value_from_dotenv(self, fresh_process, tmp_path):
        path = write_dotenv(
            tmp_path, "JOINTFM_SCHEMA_VERSION=v6\nCUDA_VISIBLE_DEVICES=1,2\n"
        )

        assert apply_gpu_selection(path) == "1,2"
        assert os.environ[CUDA_VISIBLE_DEVICES] == "1,2"

    def test_process_environment_overrides_dotenv(self, fresh_process, tmp_path):
        fresh_process.setenv(CUDA_VISIBLE_DEVICES, "3")
        path = write_dotenv(tmp_path, "CUDA_VISIBLE_DEVICES=1,2\n")

        assert apply_gpu_selection(path) == "3"
        assert os.environ[CUDA_VISIBLE_DEVICES] == "3"

    def test_missing_dotenv_selects_gpu_zero(self, fresh_process, tmp_path):
        assert apply_gpu_selection(tmp_path / ".env") == "0"
        assert os.environ[CUDA_VISIBLE_DEVICES] == "0"

    def test_dotenv_without_variable_selects_gpu_zero(self, fresh_process, tmp_path):
        path = write_dotenv(tmp_path, "JOINTFM_LOCAL_BASE_URL=http://127.0.0.1:8080\n")

        assert apply_gpu_selection(path) == "0"
        assert os.environ[CUDA_VISIBLE_DEVICES] == "0"

    def test_empty_value_hides_every_gpu(self, fresh_process, tmp_path):
        path = write_dotenv(tmp_path, "CUDA_VISIBLE_DEVICES=\n")

        assert apply_gpu_selection(path) == ""
        assert os.environ[CUDA_VISIBLE_DEVICES] == ""

    @pytest.mark.parametrize("module_name", CUDA_INITIALIZING_MODULES)
    def test_refuses_after_cuda_library_import(
        self, fresh_process, tmp_path, module_name
    ):
        fresh_process.setitem(sys.modules, module_name, types.ModuleType(module_name))
        path = write_dotenv(tmp_path, "CUDA_VISIBLE_DEVICES=1\n")

        with pytest.raises(RuntimeError, match=f"{module_name} already imported"):
            apply_gpu_selection(path)
        assert CUDA_VISIBLE_DEVICES not in os.environ

    def test_package_import_leaves_cuda_uninitialized(self):
        """The notebook cell imports the package before selecting GPUs."""
        probe = (
            "import sys\n"
            "import portfolio_optimization\n"
            "import portfolio_optimization.gpu\n"
            f"print(','.join(m for m in {CUDA_INITIALIZING_MODULES!r} if m in sys.modules))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        )

        assert result.stdout.strip() == ""

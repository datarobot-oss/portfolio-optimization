# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU selection for cuOpt, cuML, and CuPy through ``CUDA_VISIBLE_DEVICES``, and
independent jobs run in parallel across the selected GPUs."""

import multiprocessing
import multiprocessing.connection
import os
import sys
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, Union

from dotenv import dotenv_values

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
DEFAULT_CUDA_VISIBLE_DEVICES = "0"

# cvxpy creates a CUDA context while it probes its GPU solvers on import, and the
# CUDA libraries read CUDA_VISIBLE_DEVICES only when their context is created.
CUDA_INITIALIZING_MODULES = ("cvxpy", "cuopt", "cuml", "rmm")


def apply_gpu_selection(dotenv_path: Union[str, os.PathLike] = ".env") -> str:
    """Apply ``CUDA_VISIBLE_DEVICES`` from a ``.env`` file to this process.

    ``CUDA_VISIBLE_DEVICES`` is the one GPU setting for cuOpt, cuML, and CuPy: it
    lists the GPUs the process may use (one index or a comma-separated list), and
    work inside one process runs on the first of them. An empty value hides every
    GPU. The variable only takes effect if it is set before CUDA is initialized, so
    call this function before anything imports cvxpy, cuOpt, cuML, or RMM, in a
    notebook's first cell; changing the selection afterwards needs a fresh process
    (a kernel restart).

    The rules match the ``jointfm-client`` SDK and ``uv run --env-file``: a variable
    already set in the process environment overrides the one in ``.env``, and a
    missing ``.env`` file is not an error. When neither sets the variable, it is set
    to ``DEFAULT_CUDA_VISIBLE_DEVICES`` (GPU 0), so a process never spreads over
    every GPU by accident.

    Parameters
    ----------
    dotenv_path : str or os.PathLike, default ".env"
        Path of the ``.env`` file to read ``CUDA_VISIBLE_DEVICES`` from.

    Returns
    -------
    str
        The ``CUDA_VISIBLE_DEVICES`` value now in effect.

    Raises
    ------
    RuntimeError
        If cvxpy, cuOpt, cuML, or RMM is already imported, because the selection
        would then be silently ignored.

    Examples
    --------
    >>> from portfolio_optimization.gpu import apply_gpu_selection
    >>> apply_gpu_selection("../.env")  # with CUDA_VISIBLE_DEVICES=1,2 in ../.env
    '1,2'
    """
    imported = [name for name in CUDA_INITIALIZING_MODULES if name in sys.modules]
    if imported:
        raise RuntimeError(
            f"{', '.join(imported)} already imported, so CUDA may be initialized and "
            f"{CUDA_VISIBLE_DEVICES} would be ignored. Call apply_gpu_selection() "
            "before importing them, or restart the kernel."
        )
    if CUDA_VISIBLE_DEVICES not in os.environ:
        value = None
        if Path(dotenv_path).is_file():
            value = dotenv_values(dotenv_path).get(CUDA_VISIBLE_DEVICES)
        os.environ[CUDA_VISIBLE_DEVICES] = (
            DEFAULT_CUDA_VISIBLE_DEVICES if value is None else value
        )
    return os.environ[CUDA_VISIBLE_DEVICES]


def visible_gpus() -> list[str]:
    """Return the GPUs that ``CUDA_VISIBLE_DEVICES`` lists for this process.

    Returns
    -------
    list of str
        The listed entries in order (indices or GPU UUIDs, as written).

    Raises
    ------
    RuntimeError
        If ``CUDA_VISIBLE_DEVICES`` is unset, because the process would then see
        every GPU; call :func:`apply_gpu_selection` or start the script with
        ``uv run --env-file .env``.
    ValueError
        If the variable lists no GPU or lists one twice.
    """
    if CUDA_VISIBLE_DEVICES not in os.environ:
        raise RuntimeError(
            f"{CUDA_VISIBLE_DEVICES} is not set. Call apply_gpu_selection() first or "
            "start the script with `uv run --env-file .env`."
        )
    value = os.environ[CUDA_VISIBLE_DEVICES]
    gpus = [entry.strip() for entry in value.split(",") if entry.strip()]
    if not gpus:
        raise ValueError(f"{CUDA_VISIBLE_DEVICES}={value!r} lists no GPU.")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"{CUDA_VISIBLE_DEVICES}={value!r} lists a GPU twice.")
    return gpus


Result = TypeVar("Result")


@dataclass
class _Worker:
    """One spawned worker process bound to one GPU."""

    gpu: str
    process: multiprocessing.process.BaseProcess
    connection: multiprocessing.connection.Connection
    job_index: Union[int, None] = None


def run_on_gpus(
    job: Callable[..., Result], job_kwargs: Sequence[Mapping[str, Any]]
) -> list[Result]:
    """Run independent jobs in parallel, one worker process per selected GPU.

    Each element of ``job_kwargs`` is one job, run as ``job(**kwargs)``. With
    several GPUs in ``CUDA_VISIBLE_DEVICES`` (see :func:`visible_gpus`), one spawned
    worker process per GPU, up to the number of jobs, takes the next waiting job
    whenever it finishes one, so a long job does not hold back the others. Each
    worker starts with ``CUDA_VISIBLE_DEVICES`` set to its own GPU alone, so cvxpy,
    cuOpt, cuML, and CuPy in it see only that device, whatever the caller imported.
    With one GPU listed, the jobs run in sequence in the calling process, without
    worker start-up.

    Workers pay for their own imports (cvxpy and cuOpt take seconds), so the jobs
    should be whole runs, such as a rebalancing run or a scenario-generator arm,
    not sub-second solves. Jobs must not depend on each other or on state of the
    calling process: a worker receives only ``job`` and the job's keyword
    arguments.

    Parameters
    ----------
    job : callable
        A function defined at module level of an importable module, because spawned
        workers import it by name; a function defined in a notebook or in a
        script's ``__main__`` cannot be found by a worker.
    job_kwargs : sequence of mapping
        Keyword arguments of each job. Arguments and results cross process
        boundaries by pickling, so both must be picklable.

    Returns
    -------
    list
        The result of each job, in the order of ``job_kwargs``.

    Raises
    ------
    ValueError
        If ``job_kwargs`` is empty, or from :func:`visible_gpus`.
    RuntimeError
        If ``CUDA_VISIBLE_DEVICES`` is unset; if a job raises in a worker (the
        message carries the worker's traceback); or if a worker exits while
        running a job. Remaining workers are terminated before it is raised.

    Notes
    -----
    A script that calls this with several GPUs must guard its entry point with
    ``if __name__ == "__main__":``, because each spawned worker imports the
    script's main module.

    Examples
    --------
    >>> from portfolio_optimization import rebalance
    >>> from portfolio_optimization.gpu import run_on_gpus
    >>> runs = run_on_gpus(rebalance.run_rebalancing, [pct_change_run, drift_run])
    """
    if not job_kwargs:
        raise ValueError("run_on_gpus needs at least one job.")
    gpus = visible_gpus()
    if len(gpus) == 1:
        return [job(**kwargs) for kwargs in job_kwargs]

    context = multiprocessing.get_context("spawn")
    workers = [_start_worker(context, gpu, job) for gpu in gpus[: len(job_kwargs)]]
    pending = iter(enumerate(job_kwargs))
    results: dict[int, Result] = {}
    completed = False
    try:
        for worker in workers:
            _dispatch(worker, pending)
        while any(worker.job_index is not None for worker in workers):
            _collect(workers, pending, results)
        completed = True
    finally:
        _stop(workers, graceful=completed)
    return [results[index] for index in range(len(job_kwargs))]


@contextmanager
def _environment_variable(name: str, value: str) -> Iterator[None]:
    """Set an environment variable of this process for the duration of the block."""
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            del os.environ[name]
        else:
            os.environ[name] = previous


def _start_worker(
    context: multiprocessing.context.SpawnContext, gpu: str, job: Callable[..., Any]
) -> _Worker:
    """Spawn a worker that sees only ``gpu``."""
    parent_connection, child_connection = context.Pipe()
    process = context.Process(
        target=_worker_main,
        args=(job, child_connection),
        name=f"run_on_gpus-gpu{gpu}",
    )
    # A spawned child takes its environment at exec time and imports the caller's
    # main module before its target runs, so setting the variable inside the
    # target would come after a top-level `import cvxpy` had initialized CUDA.
    with _environment_variable(CUDA_VISIBLE_DEVICES, gpu):
        process.start()
    child_connection.close()
    return _Worker(gpu=gpu, process=process, connection=parent_connection)


def _worker_main(
    job: Callable[..., Any], connection: multiprocessing.connection.Connection
) -> None:
    """Run jobs received over ``connection`` until told to stop or a job fails.

    Messages in are ``(index, kwargs)`` or ``None`` to stop; messages out are
    ``(index, True, result)`` or ``(index, False, traceback_text)``. A failure is
    sent to the parent, which raises it, instead of being raised here, because an
    exception in a worker would otherwise only reach the worker's stderr.
    """
    while True:
        message = connection.recv()
        if message is None:
            return
        index, kwargs = message
        try:
            connection.send((index, True, job(**kwargs)))
        except Exception:
            connection.send((index, False, traceback.format_exc()))
            return


def _dispatch(
    worker: _Worker, pending: Iterator[tuple[int, Mapping[str, Any]]]
) -> None:
    """Send the next waiting job to an idle worker, or leave it idle if none waits."""
    item = next(pending, None)
    if item is None:
        worker.job_index = None
        return
    index, kwargs = item
    worker.job_index = index
    worker.connection.send((index, dict(kwargs)))


def _collect(
    workers: list[_Worker],
    pending: Iterator[tuple[int, Mapping[str, Any]]],
    results: dict[int, Any],
) -> None:
    """Wait for busy workers, store finished results, and hand out waiting jobs."""
    busy = [worker for worker in workers if worker.job_index is not None]
    by_connection = {worker.connection: worker for worker in busy}
    by_sentinel = {worker.process.sentinel: worker for worker in busy}
    ready = multiprocessing.connection.wait([*by_connection, *by_sentinel])
    for worker in busy:
        # A result sent just before the worker exited is still readable.
        if worker.connection in ready or worker.connection.poll():
            index, succeeded, payload = worker.connection.recv()
            if not succeeded:
                raise RuntimeError(
                    f"Job {index} failed on GPU {worker.gpu}:\n{payload}"
                )
            results[index] = payload
            _dispatch(worker, pending)
        elif worker.process.sentinel in ready:
            raise RuntimeError(
                f"Worker on GPU {worker.gpu} exited with code "
                f"{worker.process.exitcode} while running job {worker.job_index}."
            )


def _stop(workers: list[_Worker], graceful: bool) -> None:
    """Stop every worker: ask idle ones to exit, or terminate all after a failure."""
    for worker in workers:
        if graceful:
            worker.connection.send(None)
        elif worker.process.is_alive():
            worker.process.terminate()
    for worker in workers:
        worker.process.join()
        worker.connection.close()

# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The results table: one Parquet file holding the rows and the run's identity.

The identity (data, experiment parameters, arms, measurements, host, package
versions, and JointFM services) is stored in the file's schema metadata, so
the rows never travel without the conditions they were measured under. A run
that resumes into an existing file must have the same identity, except that it
may add arms; the command lines of its sessions accumulate.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

METADATA_KEY = b"simulator_comparison"
# One arm's call at one origin writes the rows of every horizon at once, so a
# resumed run skips the rows of a (trial, origin, arm) as a whole.
CALL_COLUMNS = ("trial", "origin", "arm")
KEY_COLUMNS = (*CALL_COLUMNS, "horizon")
# Identity entries keyed by arm name, which later sessions may extend.
ACCUMULATING_KEYS = ("arms", "services")


@dataclass(frozen=True)
class RunMetadata:
    """What a results file records beside its rows.

    Attributes
    ----------
    identity : dict
        Everything that must not change while rows accumulate; JSON-ready.
    commands : list of list of str
        Command line of every session that wrote to the file, oldest first.
    """

    identity: dict
    commands: list[list[str]]


def read_results(path: Path) -> tuple[pd.DataFrame, RunMetadata]:
    """Read a results file written by ``write_results``.

    Raises
    ------
    ValueError
        If the file carries no simulator-comparison metadata.
    """
    table = pq.read_table(path)
    metadata = table.schema.metadata or {}
    if METADATA_KEY not in metadata:
        raise ValueError(f"{path} is not a simulator-comparison results file.")
    payload = json.loads(metadata[METADATA_KEY])
    return table.to_pandas(), RunMetadata(**payload)


def write_results(path: Path, frame: pd.DataFrame, metadata: RunMetadata) -> None:
    """Write rows and metadata, replacing ``path`` atomically.

    The file is written beside ``path`` and moved over it, so an interrupted
    run leaves the previous complete file instead of a truncated one.
    """
    table = pa.Table.from_pandas(frame, preserve_index=False)
    payload = json.dumps(
        {"identity": metadata.identity, "commands": metadata.commands}
    ).encode()
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), METADATA_KEY: payload}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    pq.write_table(table, partial)
    os.replace(partial, path)


def merge_identity(path: Path, stored: dict, current: dict) -> dict:
    """Combine a file's identity with this session's, refusing other conditions.

    Every entry must match, except the per-arm entries in ``ACCUMULATING_KEYS``:
    a session may add arms the file does not hold yet (the JointFM arm on the
    other device), while an arm it already holds must keep its settings and
    service.

    Returns
    -------
    dict
        The stored identity with this session's new arms added.

    Raises
    ------
    ValueError
        Naming every entry that differs.
    """
    fixed = (stored.keys() | current.keys()) - set(ACCUMULATING_KEYS)
    differing = {
        key: (stored.get(key), current.get(key))
        for key in sorted(fixed)
        if stored.get(key) != current.get(key)
    }
    for key in ACCUMULATING_KEYS:
        for arm in sorted(stored[key].keys() & current[key].keys()):
            if stored[key][arm] != current[key][arm]:
                differing[f"{key}.{arm}"] = (stored[key][arm], current[key][arm])
    if differing:
        details = "; ".join(
            f"{key}: stored {old!r}, now {new!r}"
            for key, (old, new) in differing.items()
        )
        raise ValueError(
            f"{path} was measured under other conditions, so its rows cannot be "
            f"extended. Write to a new file or restore the conditions. {details}"
        )
    merged = dict(stored)
    for key in ACCUMULATING_KEYS:
        merged[key] = {**stored[key], **current[key]}
    return merged

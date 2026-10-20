# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired comparison of the scenario simulators on identical inputs.

Every arm (one simulator on one device) receives the same assets, look-back
window, and scenario seed at every forecast origin of every trial, so arms
differ only in the simulator. One run records, per (trial, origin, arm), the
wall time of the scenario-generation call and every registered measurement of
its scenarios. See ``README.md`` in this directory for how to run it.
"""

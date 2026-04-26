# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Causalrepair Environment.

The CausalRepair environment is a simple test environment that echoes back messages.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, Literal, Optional


class CausalrepairAction(BaseModel):
    """Action for the Causalrepair environment."""
    action_type: Literal["diagnose", "intervene", "propagate", "commit_repair"]
    target: Optional[str] = None  # e.g. "line_42", "function_foo", "test_bar"
    value: Optional[Any] = None  # new value / edit for intervene
    rationale: Optional[str] = None  # text justification for commit_repair
    payload: Dict[str, Any] = Field(default_factory=dict)


class CausalrepairObservation(BaseModel):
    """Observation from the Causalrepair environment."""
    description: str
    extra: Dict[str, Any] = Field(default_factory=dict)
    diagnose_result: Optional[str] = None


class StepResult(BaseModel):
    observation: CausalrepairObservation
    reward: float
    done: bool
    info: dict

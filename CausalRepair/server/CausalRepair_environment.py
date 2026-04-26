# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Causalrepair Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""


from openenv.core.env_server.interfaces import Environment
from typing import Any, Dict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CausalrepairAction, CausalrepairObservation, StepResult
from pydantic import BaseModel, Field


class CausalrepairEnvironment(Environment):
    """
    CausalRepair RL environment for diagnosis and repair using a pluggable adapter.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    @property
    def state(self):
            return {
                "world": self.world,
                "steps": self.steps,
                "diagnose_calls": self.diagnose_calls,
                "done": self._done,
            }
    def __init__(self, adapter, max_steps: int = 10, diagnose_budget: int = 3):
        self.adapter = adapter
        self.max_steps = max_steps
        self.diagnose_budget = diagnose_budget
        self.steps = 0
        self.diagnose_calls = 0
        self._done = False
        self.reset()

    def reset(self):
        self.world = self.adapter.generate_world()
        self.adapter.inject_fault(self.world)
        self.steps = 0
        self.diagnose_calls = 0
        self._done = False
        obs = self.adapter.render_observation(self.world)
        return StepResult(
            observation=obs.model_dump(),
            reward=0.0,
            done=False,
            info={"steps": 0, "diagnose_calls": 0},
        )

    def step(self, action: CausalrepairAction):
        self.steps += 1
        done = False
        action_type = action.action_type

        if action_type == "diagnose":
            self.diagnose_calls += 1
            self.adapter.diagnose(self.world, action.target or "")
        elif action_type == "intervene":
            self.adapter.intervene(self.world, action.target, action.value)
        elif action_type == "propagate":
            self.adapter.propagate(self.world)
        elif action_type == "commit_repair":
            done = True

        if not done and self.steps >= self.max_steps:
            done = True

        obs = self.adapter.render_observation(self.world)
        self._done = done

        info = {
            "steps": self.steps,
            "diagnose_calls": self.diagnose_calls,
            "constraints_ok": self.adapter.check_constraints(self.world),
            "action_type": action_type,
        }
        return StepResult(
            observation=obs.model_dump(),
            reward=0.0,
            done=done,
            info=info,
        )

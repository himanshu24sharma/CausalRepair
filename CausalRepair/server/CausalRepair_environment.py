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
        obs = self.adapter.render_observation(self.world)
        obs_dict = obs.model_dump()
        # If a diagnose_result was recently set, include it in the state observation
        if hasattr(self, "_last_diagnose_result") and self._last_diagnose_result is not None:
            obs_dict["diagnose_result"] = self._last_diagnose_result
        return {
            "world": self.world,
            "steps": self.steps,
            "diagnose_calls": self.diagnose_calls,
            "done": self._done,
            "observation": obs_dict
        }
    def __init__(self, adapter, max_steps=20):
        self.adapter = adapter
        print(f"Using adapter: {adapter.__class__.__name__}")
        self.max_steps = max_steps
        self.prev_observation = []
        self.reset()

    def reset(self):
        print("Environment reset called")
        self.world = self.adapter.generate_world()
        self.adapter.inject_fault(self.world)
        self.steps = 0
        self.diagnose_calls = 0
        self._done = False
        obs = self.adapter.render_observation(self.world)
        obs_dict = obs.model_dump()
        obs_dict["prev_observation"] = self.prev_observation.copy()
        # Only keep the previous observation, not including the current one
        if self.prev_observation:
            self.prev_observation = [self.prev_observation[-1]]
        else:
            self.prev_observation = []
        self.prev_observation.append(obs_dict.copy())
        print(f"[DEBUG] Initial observation: {obs_dict}", flush=True)
        return StepResult(
            observation=obs_dict,
            reward=0.0,
            done=False,
            info={"steps": self.steps, "diagnose_calls": self.diagnose_calls}
        )

    def step(self, action: CausalrepairAction):
        reward = 0.0
        done = False
        info = {"steps": self.steps, "diagnose_calls": self.diagnose_calls}
        diagnose_results = []
        if action.action_type == "diagnose":
            self.diagnose_calls += 1
            diagnose_result = self.adapter.diagnose(self.world, action.target)
            self._last_diagnose_result = diagnose_result
            self.adapter.propagate(self.world)
            reward = 0.0 if self.diagnose_calls <= 3 else -0.1
            diagnose_results.append(diagnose_result)
        elif action.action_type == "intervene":
            self.adapter.intervene(self.world, action.target, action.value)
            reward = 0.0
        elif action.action_type == "propagate":
            self.adapter.propagate(self.world)
            reward = 0.0
        elif action.action_type == "commit_repair":
            success = self.adapter.check_constraints(self.world)
            reward = 1.0 if success else -0.5
            reward += 0.3 * (1 - self.steps / self.max_steps)
            if self.diagnose_calls <= 3:
                reward += 0.2
            done = True
            self._done = True
        else:
            reward = -0.1
        self.steps += 1
        info = {"steps": self.steps, "diagnose_calls": self.diagnose_calls}
        obs = self.adapter.render_observation(self.world)
        sys.stdout.flush()
        obs_dict = obs.model_dump()
        # If diagnose_result exists, add it to the observation
        print(diagnose_result)
        if action.action_type == "diagnose":
            if "diagnose_results" not in obs_dict:
                obs_dict["diagnose_results"] = []
            obs_dict["diagnose_results"].append(diagnose_result)
        print(obs_dict.get("diagnose_results", []))
        return StepResult(
            observation=obs_dict,
            reward=reward,
            done=done,
            info=info
        )

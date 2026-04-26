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
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CausalrepairAction, StepResult

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
    def __init__(self, adapter, max_steps=20, diagnose_budget=3):
        self.adapter = adapter
        print(f"Using adapter: {adapter.__class__.__name__}")
        self.max_steps = max_steps
        self.diagnose_budget = diagnose_budget
        self.steps = 0
        self.diagnose_calls = 0
        self._done = False
        self.prev_observation = []
        self._last_diagnose_result = None
        self.reset()

    def reset(self):
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
            info={"steps": 0, "diagnose_calls": 0},
        )

    def step(self, action: CausalrepairAction):
        self.steps += 1
        done = False
        action_type = action.action_type
        if action_type == "diagnose":
            self.diagnose_calls += 1
            diagnose_result = self.adapter.diagnose(self.world, action.target)
            self._last_diagnose_result = diagnose_result
            self.adapter.propagate(self.world)
        elif action_type == "intervene":
            self.adapter.intervene(self.world, action.target, action.value)
        elif action_type == "propagate":
            self.adapter.propagate(self.world)
        elif action_type == "commit_repair":
            done = True

        if not done and self.steps >= self.max_steps:
            done = True

        obs = self.adapter.render_observation(self.world)
        sys.stdout.flush()
        obs_dict = obs.model_dump()
        # If diagnose_result exists, add it to the observation
        if action.action_type == "diagnose":
            if "diagnose_results" not in obs_dict:
                obs_dict["diagnose_results"] = []
            obs_dict["diagnose_results"].append(diagnose_result)
        self._done = done

        info = {
            "steps": self.steps,
            "diagnose_calls": self.diagnose_calls,
            "constraints_ok": self.adapter.check_constraints(self.world),
            "action_type": action_type,
        }
        return StepResult(
            observation=obs_dict,
            reward=0.0,
            done=done,
            info=info,
        )

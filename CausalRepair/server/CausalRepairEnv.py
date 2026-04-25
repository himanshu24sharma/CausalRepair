# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
CausalRepairEnv: Core environment for causal diagnosis and repair (OpenEnv-compatible).
"""
from typing import Any, Dict
from .. import apiContact

class CausalRepairEnv:
    def __init__(self, adapter, max_steps=20):
        self.adapter = adapter
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.world = self.adapter.generate_world()
        self.adapter.inject_fault(self.world)
        self.steps = 0
        self.diagnose_calls = 0
        return self.adapter.render_observation(self.world)

    def step(self, action: apiContact.Action):
        reward = 0.0
        done = False
        info = {"steps": self.steps, "diagnose_calls": self.diagnose_calls}

        if action.action_type == "diagnose":
            self.diagnose_calls += 1
            self.adapter.diagnose(self.world, action.target)
            reward = 0.0 if self.diagnose_calls <= 3 else -0.1
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
        else:
            reward = -0.1
        self.steps += 1
        info = {"steps": self.steps, "diagnose_calls": self.diagnose_calls}
        obs = self.adapter.render_observation(self.world)
        return obs, reward, done, info

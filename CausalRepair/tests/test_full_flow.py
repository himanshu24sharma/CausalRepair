"""
Integration test: full repair episode with a scripted perfect agent.

Covers:
  - CausalrepairEnvironment + CodeRepairAdapter wiring
  - reset() injects a fault; check_constraints flips False -> True
  - step() correctly handles all four action types
  - compute_reward accumulates shaped reward matching the spec
"""
import math
import sys
from pathlib import Path

# Ensure package root is on sys.path when run directly (conftest.py handles pytest)
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from models import CausalrepairAction
from server.CausalRepair_environment import CausalrepairEnvironment
from server.code_repair_adapter import CodeRepairAdapter
from inference import compute_reward

# Must match _HEALTHY_SOURCES["add"] in code_repair_adapter.py exactly
CORRECT_ADD_SRC = "def add(x, y):\n    return x + y\n"


def test_full_flow_scripted_agent_code_domain():
    # 1. Build env with code adapter
    env = CausalrepairEnvironment(
        adapter=CodeRepairAdapter(),
        max_steps=10,
        diagnose_budget=3,
    )

    # 2. Reset and check initial state
    first = env.reset()
    assert first.reward == 0.0
    assert first.done is False

    # At reset, a fault has been injected — at least one constraint must be failing
    assert env.adapter.check_constraints(env.world) is False

    total_reward = 0.0

    # 3. Scripted "perfect" repair policy:
    #    diagnose -> intervene (fix add) -> propagate -> commit_repair
    actions = [
        CausalrepairAction(action_type="diagnose", target="add"),
        CausalrepairAction(action_type="intervene", target="add", value=CORRECT_ADD_SRC),
        CausalrepairAction(action_type="propagate"),
        CausalrepairAction(
            action_type="commit_repair",
            target="add",
            value=CORRECT_ADD_SRC,
            rationale="restore add correctness",
        ),
    ]

    last_result = first

    for action in actions:
        result = env.step(action)
        reward = compute_reward(
            action=action,
            done=result.done,
            info=result.info,
            max_steps=env.max_steps,
            diagnose_budget=env.diagnose_budget,
        )
        total_reward += reward
        last_result = result

        if result.done:
            break

    # 4. After the scripted sequence, constraints must be satisfied and episode done
    assert env.adapter.check_constraints(env.world) is True
    assert last_result.done is True

    # 5. Verify the shaped reward matches the formula exactly:
    #
    #   diagnose:      -0.05
    #   intervene:      0.0
    #   propagate:      0.0
    #   commit_repair:  1.0 (success)
    #                    + 0.3 * (1 - steps/max_steps)   efficiency bonus (4 steps)
    #                    + 0.2                            budget bonus (1 diagnose <= 3)
    #
    expected_commit = 1.0 + 0.3 * (1 - 4 / env.max_steps) + 0.2
    expected_total = -0.05 + 0.0 + 0.0 + expected_commit

    assert math.isclose(total_reward, expected_total, rel_tol=1e-6), (
        f"total_reward={total_reward}, expected={expected_total}"
    )

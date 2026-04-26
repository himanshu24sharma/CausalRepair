"""
Tests for CausalrepairEnvironment behavior (reward-free).

Validates that the environment correctly manages world state, step counters,
and the done flag across the full episode lifecycle. Reward is always 0.0
from the env; shaped reward is tested separately in test_compute_reward.py.

Runs two ways:
    1. As a plain script:   python tests/test_env_behavior.py
    2. With pytest:         pytest tests/test_env_behavior.py -v

Run from inside the package root: c:\\Users\\hp\\CausalRepair\\CausalRepair
"""
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models import CausalrepairAction
from server.code_repair_adapter import CodeRepairAdapter
from server.CausalRepair_environment import CausalrepairEnvironment

# The correct source for the "add" function — what inject_fault mutates away from.
CORRECT_ADD_SRC = "def add(x, y):\n    return x + y\n"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reset_creates_faulty_world():
    """reset() must return a clean StepResult and leave the world with a violation."""
    adapter = CodeRepairAdapter()
    env = CausalrepairEnvironment(adapter=adapter)

    result = env.reset()

    assert result.reward == 0.0
    assert result.done is False
    assert result.info["steps"] == 0
    assert result.info["diagnose_calls"] == 0
    # inject_fault is called by reset(), so at least one constraint should fail
    assert adapter.check_constraints(env.world) is False


def test_diagnose_increments_counters_and_preserves_constraints():
    """A diagnose step increments counters but must not alter world constraints."""
    adapter = CodeRepairAdapter()
    env = CausalrepairEnvironment(adapter=adapter)
    env.reset()

    step_result = env.step(CausalrepairAction(action_type="diagnose", target="add"))

    assert env.steps == 1
    assert env.diagnose_calls == 1
    assert step_result.reward == 0.0
    # diagnose is read-only; the fault must still be present
    assert adapter.check_constraints(env.world) is False


def test_intervene_and_propagate_can_fix_constraints():
    """A diagnose → intervene → propagate sequence should make all constraints pass."""
    adapter = CodeRepairAdapter()
    env = CausalrepairEnvironment(adapter=adapter)
    env.reset()

    env.step(CausalrepairAction(action_type="diagnose", target="add"))
    env.step(CausalrepairAction(action_type="intervene", target="add", value=CORRECT_ADD_SRC))
    env.step(CausalrepairAction(action_type="propagate"))

    assert adapter.check_constraints(env.world) is True
    assert env.steps >= 3


def test_commit_marks_done():
    """commit_repair must set done=True and flip env._done after a successful repair."""
    adapter = CodeRepairAdapter()
    env = CausalrepairEnvironment(adapter=adapter)
    env.reset()

    # Repair the world first so commit_repair is meaningful
    env.step(CausalrepairAction(action_type="diagnose", target="add"))
    env.step(CausalrepairAction(action_type="intervene", target="add", value=CORRECT_ADD_SRC))
    env.step(CausalrepairAction(action_type="propagate"))

    res = env.step(CausalrepairAction(
        action_type="commit_repair",
        target="add",
        value=CORRECT_ADD_SRC,
        rationale="restore add correctness",
    ))

    assert res.done is True
    assert env._done is True


def test_timeout_triggers_done():
    """When steps reach max_steps on a non-commit action the env must set done=True."""
    adapter = CodeRepairAdapter()
    # max_steps=3: steps 1 and 2 are below the limit; step 3 hits the boundary
    env = CausalrepairEnvironment(adapter=adapter, max_steps=3)
    env.reset()

    r1 = env.step(CausalrepairAction(action_type="diagnose", target="add"))
    assert r1.done is False

    r2 = env.step(CausalrepairAction(action_type="diagnose", target="add"))
    assert r2.done is False

    # Step 3: self.steps becomes 3 which equals max_steps → timeout
    r3 = env.step(CausalrepairAction(action_type="propagate"))
    assert r3.done is True


# ---------------------------------------------------------------------------
# Script runner (mirrors test_code_adapter.py pattern)
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_reset_creates_faulty_world,
    test_diagnose_increments_counters_and_preserves_constraints,
    test_intervene_and_propagate_can_fix_constraints,
    test_commit_marks_done,
    test_timeout_triggers_done,
]


def _run_all_as_script():
    passed = 0
    failed = 0
    for fn in ALL_TESTS:
        name = fn.__name__
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"  PASS  {name}")
    print(f"\n{passed} passed, {failed} failed (out of {len(ALL_TESTS)})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all_as_script())

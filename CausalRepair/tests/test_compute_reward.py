"""
Unit tests for compute_reward() in inference.py.

Tests the shaped reward function in complete isolation from the environment:
all inputs (action, done, info, max_steps, diagnose_budget) are constructed
directly, so these tests never touch the adapter or world state.

Runs two ways:
    1. As a plain script:   python tests/test_compute_reward.py
    2. With pytest:         pytest tests/test_compute_reward.py -v

Run from inside the package root: c:\\Users\\hp\\CausalRepair\\CausalRepair
"""
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models import CausalrepairAction
from inference import compute_reward


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def action(action_type: str, **kwargs) -> CausalrepairAction:
    """Shorthand for building a CausalrepairAction in tests."""
    return CausalrepairAction(action_type=action_type, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reward_diagnose_penalty():
    """Each diagnose call should incur a small -0.05 penalty."""
    a = action("diagnose", target="add")
    info = {"constraints_ok": False, "steps": 1, "diagnose_calls": 1}
    reward = compute_reward(a, done=False, info=info, max_steps=10, diagnose_budget=3)
    assert reward == -0.05


def test_reward_intervene_zero():
    """intervene returns 0.0; the main reward is deferred to commit."""
    a = action("intervene", target="add", value="def add(x, y):\n    return x + y\n")
    info = {"constraints_ok": False, "steps": 2, "diagnose_calls": 1}
    reward = compute_reward(a, done=False, info=info, max_steps=10, diagnose_budget=3)
    assert reward == 0.0


def test_reward_propagate_zero():
    """propagate also returns 0.0 mid-episode."""
    a = action("propagate")
    info = {"constraints_ok": False, "steps": 3, "diagnose_calls": 1}
    reward = compute_reward(a, done=False, info=info, max_steps=10, diagnose_budget=3)
    assert reward == 0.0


def test_reward_successful_commit_with_efficiency_and_budget_bonus():
    """
    Successful commit: 1.0 + efficiency_bonus + budget_bonus.

    With steps=4, max_steps=20, diagnose_calls=2, diagnose_budget=3:
        efficiency = 0.3 * (1 - 4/20) = 0.3 * 0.8 = 0.24
        budget_bonus = 0.2   (diagnose_calls 2 <= budget 3)
        total = 1.44
    """
    a = action("commit_repair", target="add", value="def add(x, y):\n    return x + y\n")
    info = {"constraints_ok": True, "steps": 4, "diagnose_calls": 2}
    reward = compute_reward(a, done=True, info=info, max_steps=20, diagnose_budget=3)
    expected = 1.0 + 0.3 * (1 - 4 / 20) + 0.2
    assert abs(reward - expected) < 1e-6


def test_reward_successful_commit_over_diagnose_budget_loses_bonus():
    """When diagnose_calls exceeds budget the 0.2 bonus is forfeited."""
    a = action("commit_repair", target="add")
    info = {"constraints_ok": True, "steps": 4, "diagnose_calls": 5}
    reward = compute_reward(a, done=True, info=info, max_steps=20, diagnose_budget=3)
    expected = 1.0 + 0.3 * (1 - 4 / 20)  # no budget_bonus
    assert abs(reward - expected) < 1e-6


def test_reward_failed_commit_negative():
    """Committing while constraints are still broken must return -1.0."""
    a = action("commit_repair", target="add", value="def add(x, y):\n    return x - y\n")
    info = {"constraints_ok": False, "steps": 4, "diagnose_calls": 2}
    reward = compute_reward(a, done=True, info=info, max_steps=20, diagnose_budget=3)
    assert reward == -1.0


def test_reward_timeout_penalty():
    """Episode ended via timeout (non-commit done=True) must return -0.5."""
    a = action("propagate")
    info = {"constraints_ok": False, "steps": 10, "diagnose_calls": 2}
    reward = compute_reward(a, done=True, info=info, max_steps=10, diagnose_budget=3)
    assert reward == -0.5


# ---------------------------------------------------------------------------
# Script runner (mirrors test_code_adapter.py pattern)
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_reward_diagnose_penalty,
    test_reward_intervene_zero,
    test_reward_propagate_zero,
    test_reward_successful_commit_with_efficiency_and_budget_bonus,
    test_reward_successful_commit_over_diagnose_budget_loses_bonus,
    test_reward_failed_commit_negative,
    test_reward_timeout_penalty,
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

"""
Smoke tests for Step 2: CodeRepairAdapter (hardcoded V1).

Runs two ways:
    1. As a plain script:   python tests/test_code_adapter.py
    2. With pytest:         pytest tests/test_code_adapter.py -v

Run from inside the package root: c:\\Users\\hp\\CausalRepair\\CausalRepair
"""
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models import CausalrepairObservation
from server.base_adapter import BaseAdapter
from server.code_repair_adapter import CodeRepairAdapter


def test_inherits_from_base_adapter():
    assert issubclass(CodeRepairAdapter, BaseAdapter)
    assert CodeRepairAdapter().domain_name == "code"


def test_generate_world_is_green():
    """Fresh world must satisfy all constraints (all 3 tests pass)."""
    adapter = CodeRepairAdapter()
    world = adapter.generate_world()
    assert set(world["entities"]) == {"add", "sub", "mul"}
    assert set(world["tests"]) == {"test_add", "test_sub", "test_mul"}
    assert adapter.check_constraints(world) is True
    assert all(t["status"] == "pass" for t in world["tests"].values())


def test_inject_fault_breaks_at_least_one_constraint():
    adapter = CodeRepairAdapter()
    world = adapter.generate_world()
    adapter.inject_fault(world)
    assert "x - y" in world["entities"]["add"], "inject_fault should mutate add in-place"
    assert world["meta"]["fault_ground_truth"] is not None
    assert world["meta"]["fault_ground_truth"]["entity"] == "add"
    assert adapter.check_constraints(world) is False
    assert world["tests"]["test_add"]["status"] == "fail"
    assert world["tests"]["test_sub"]["status"] == "pass"
    assert world["tests"]["test_mul"]["status"] == "pass"


def test_intervene_with_correct_source_restores_green():
    adapter = CodeRepairAdapter()
    world = adapter.generate_world()
    adapter.inject_fault(world)
    assert adapter.check_constraints(world) is False

    adapter.intervene(world, "add", "def add(x, y):\n    return x + y\n")
    for t in world["tests"].values():
        assert t["status"] == "unknown", "intervene should mark tests stale"

    adapter.propagate(world)
    assert adapter.check_constraints(world) is True


def test_intervene_with_wrong_source_keeps_constraint_violated():
    adapter = CodeRepairAdapter()
    world = adapter.generate_world()
    adapter.inject_fault(world)
    adapter.intervene(world, "add", "def add(x, y):\n    return x * y\n")
    adapter.propagate(world)
    assert adapter.check_constraints(world) is False


def test_intervene_with_unparseable_source_marks_test_failing():
    """Garbage source should not crash propagate; the test that touches it fails."""
    adapter = CodeRepairAdapter()
    world = adapter.generate_world()
    adapter.intervene(world, "add", "this is not python code at all")
    adapter.propagate(world)
    assert world["tests"]["test_add"]["status"] == "fail"
    assert "SyntaxError" in world["tests"]["test_add"]["msg"]
    assert world["tests"]["test_sub"]["status"] == "pass"


def test_diagnose_includes_source_and_failing_tests():
    adapter = CodeRepairAdapter()
    world = adapter.generate_world()
    adapter.inject_fault(world)
    out = adapter.diagnose(world, "add")
    assert isinstance(out, str)
    assert "add" in out
    assert "x - y" in out
    assert "test_add" in out


def test_diagnose_unknown_entity_is_safe():
    adapter = CodeRepairAdapter()
    world = adapter.generate_world()
    out = adapter.diagnose(world, "nonexistent_function")
    assert "unknown entity" in out


def test_render_observation_returns_correct_shape_and_format():
    adapter = CodeRepairAdapter()
    world = adapter.generate_world()
    obs = adapter.render_observation(world)
    assert isinstance(obs, CausalrepairObservation)
    assert isinstance(obs.description, str)
    text = obs.description
    for required_section in ("DOMAIN:", "STATE:", "RULES:", "CONSTRAINTS:", "AVAILABLE ACTIONS:"):
        assert required_section in text, f"observation missing {required_section!r}"
    for action_keyword in ('diagnose("entity_name")', 'intervene("entity_name", "value")', "propagate()", "commit_repair("):
        assert action_keyword in text, f"observation missing action keyword {action_keyword!r}"


def test_render_observation_marks_violated_after_fault():
    adapter = CodeRepairAdapter()
    world = adapter.generate_world()
    adapter.inject_fault(world)
    text = adapter.render_observation(world).description
    assert "[VIOLATED] test_add" in text
    assert "[OK] test_sub" in text
    assert "[OK] test_mul" in text
    assert "add: FAILING" in text
    assert "sub: PASSING" in text


def test_full_repair_episode_roundtrip():
    """End-to-end: generate -> fault -> diagnose -> intervene -> propagate -> commit OK."""
    adapter = CodeRepairAdapter()
    world = adapter.generate_world()
    adapter.inject_fault(world)
    assert not adapter.check_constraints(world)
    diag = adapter.diagnose(world, "add")
    assert "x - y" in diag
    adapter.intervene(world, "add", "def add(x, y):\n    return x + y\n")
    adapter.propagate(world)
    assert adapter.check_constraints(world)


ALL_TESTS = [
    test_inherits_from_base_adapter,
    test_generate_world_is_green,
    test_inject_fault_breaks_at_least_one_constraint,
    test_intervene_with_correct_source_restores_green,
    test_intervene_with_wrong_source_keeps_constraint_violated,
    test_intervene_with_unparseable_source_marks_test_failing,
    test_diagnose_includes_source_and_failing_tests,
    test_diagnose_unknown_entity_is_safe,
    test_render_observation_returns_correct_shape_and_format,
    test_render_observation_marks_violated_after_fault,
    test_full_repair_episode_roundtrip,
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

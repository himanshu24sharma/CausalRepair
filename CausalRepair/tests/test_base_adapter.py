"""
Smoke tests for Step 1: BaseAdapter ABC + MockAdapter conformance.

WHAT THIS FILE TESTS:
    BaseAdapter is an Abstract Base Class (ABC). That means:
      - You CANNOT instantiate BaseAdapter() directly — it raises TypeError.
      - Any subclass that forgets to implement even ONE of the 7 methods
        also raises TypeError when you try to instantiate it.
      - MockAdapter (the hydraulic domain) implements all 7 methods, so it
        instantiates fine and its round-trip behaviour is verified.

    Think of these tests as "does the contract exist and does it enforce itself?"

HOW TO RUN:
    Option 1 — plain Python script (no pytest needed):
        cd c:\\Users\\hp\\CausalRepair\\CausalRepair
        ..\\venv\\Scripts\\python.exe tests\\test_base_adapter.py

    Option 2 — with pytest (after `pip install pytest`):
        cd c:\\Users\\hp\\CausalRepair\\CausalRepair
        ..\\venv\\Scripts\\pytest.exe tests\\test_base_adapter.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure imports work when this file is run as a plain script.
# conftest.py does the same thing for pytest runs.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models import CausalrepairObservation
from server.base_adapter import BaseAdapter
from server.mock_adapter import MockAdapter


# The exact set of methods Person B's CausalRepairEnv calls on the adapter.
# If any of these are missing from BaseAdapter's abstract methods, the test fails.
REQUIRED_METHODS = {
    "generate_world",    # build a fresh healthy world
    "inject_fault",      # silently break one thing
    "render_observation",# describe the world as text for the LLM
    "diagnose",          # reveal info about one entity
    "intervene",         # apply a proposed fix
    "propagate",         # re-run rules/tests after a change
    "check_constraints", # did all constraints pass?
}


# ---------------------------------------------------------------------------
# Tests for the ABC itself
# ---------------------------------------------------------------------------

def test_base_adapter_is_abstract():
    """
    BaseAdapter() directly must raise TypeError.

    This is the core ABC property: you can never use BaseAdapter directly,
    only through a concrete subclass that implements all 7 methods.
    """
    raised = False
    try:
        BaseAdapter()
    except TypeError:
        raised = True
    assert raised, "BaseAdapter() should raise TypeError because it has abstract methods"


def test_base_adapter_declares_all_seven_methods():
    """
    BaseAdapter must declare exactly the 7 abstract methods Person B's env calls.

    Why this matters: if we add or remove a method from the ABC, Person B's
    CausalRepairEnv will silently call a method that doesn't exist (AttributeError
    at runtime). This test catches that mismatch at import time instead.
    """
    declared = set(BaseAdapter.__abstractmethods__)
    missing = REQUIRED_METHODS - declared  # things env needs but ABC doesn't enforce
    extra = declared - REQUIRED_METHODS    # things ABC enforces but env doesn't call
    assert not missing, f"BaseAdapter is missing abstract methods: {missing}"
    assert not extra, f"BaseAdapter has unexpected abstract methods: {extra}"


# ---------------------------------------------------------------------------
# Tests for MockAdapter (the hydraulic reference implementation)
# ---------------------------------------------------------------------------

def test_mock_adapter_inherits_from_base():
    """MockAdapter must be a subclass of BaseAdapter to lock the contract."""
    assert issubclass(MockAdapter, BaseAdapter), (
        "MockAdapter must inherit from BaseAdapter to lock the contract"
    )


def test_mock_adapter_instantiates():
    """
    MockAdapter implements all 7 methods, so MockAdapter() must succeed.
    Also checks domain_name is set (not the default "unknown").
    """
    adapter = MockAdapter()
    assert isinstance(adapter, BaseAdapter)
    assert adapter.domain_name == "hydraulic"


def test_mock_adapter_round_trip_healthy_world():
    """
    generate_world() must return a healthy world where check_constraints() is True.

    For the hydraulic domain:
      - valve=open, pressure=50, alarm=False
      - constraint: alarm must be False (alarm=False means no alert)
    """
    adapter = MockAdapter()
    world = adapter.generate_world()
    assert world == {"valve": "open", "pressure": 50, "alarm": False}
    assert adapter.check_constraints(world) is True


def test_mock_adapter_round_trip_fault_breaks_constraint():
    """
    inject_fault() then propagate() must flip at least one constraint to violated.

    inject_fault sets valve=closed (in-place, returns None).
    propagate() updates pressure and alarm based on the new valve state.
    Result: alarm=True, but constraint says alarm must be False -> violated.
    """
    adapter = MockAdapter()
    world = adapter.generate_world()

    adapter.inject_fault(world)
    # inject_fault is in-place — it modifies the dict we passed in
    assert world["valve"] == "closed", "inject_fault should mutate in-place"

    adapter.propagate(world)
    assert world["pressure"] == 90      # valve closed -> high pressure
    assert world["alarm"] is True       # high pressure -> alarm triggered
    assert adapter.check_constraints(world) is False, (
        "constraint must be violated after fault propagation"
    )


def test_mock_adapter_round_trip_intervene_restores_constraint():
    """
    intervene() back to the healthy value + propagate() must restore constraint.

    This is the full repair loop:
        inject_fault -> (LLM diagnoses) -> intervene(fix) -> propagate -> check_constraints=True
    """
    adapter = MockAdapter()
    world = adapter.generate_world()
    adapter.inject_fault(world)
    adapter.propagate(world)
    assert adapter.check_constraints(world) is False  # broken

    # LLM intervenes: set valve back to "open" (the correct value)
    adapter.intervene(world, "valve", "open")
    adapter.propagate(world)
    assert adapter.check_constraints(world) is True   # fixed


def test_mock_adapter_render_observation_returns_pydantic_model():
    """
    render_observation() must return a CausalrepairObservation, not a bare string.

    CausalrepairObservation is the Pydantic model in models.py with two fields:
      - description (str): the 5-section text the LLM reads
      - extra (dict):      structured data for logging
    Person B's env expects this type, not a raw string.
    """
    adapter = MockAdapter()
    world = adapter.generate_world()
    obs = adapter.render_observation(world)
    assert isinstance(obs, CausalrepairObservation)
    assert isinstance(obs.description, str)
    assert "valve=" in obs.description   # MockAdapter puts entity values in description
    assert isinstance(obs.extra, dict)


def test_mock_adapter_diagnose_returns_string():
    """
    diagnose() must return a plain string (not a Pydantic model).
    The string is appended to the LLM's context as a tool-call result.
    """
    adapter = MockAdapter()
    world = adapter.generate_world()
    out = adapter.diagnose(world, "valve")
    assert isinstance(out, str)
    assert "valve" in out and "open" in out


def test_partial_subclass_still_abstract():
    """
    A subclass that implements only SOME of the 7 methods must still fail to instantiate.

    This proves the ABC enforcement works for future adapter authors:
    if you forget to implement e.g. propagate(), Python will catch it at
    instantiation time rather than at the first env.step() call.
    """
    # Intentionally incomplete: only generate_world is implemented
    class IncompleteAdapter(BaseAdapter):
        domain_name = "broken"

        def generate_world(self):
            return {}

    raised = False
    try:
        IncompleteAdapter()  # should raise TypeError: missing 6 abstract methods
    except TypeError:
        raised = True
    assert raised, (
        "Subclasses missing any of the 7 abstract methods must fail to instantiate"
    )


# ---------------------------------------------------------------------------
# Script-mode runner (runs when file is executed directly, not via pytest)
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_base_adapter_is_abstract,
    test_base_adapter_declares_all_seven_methods,
    test_mock_adapter_inherits_from_base,
    test_mock_adapter_instantiates,
    test_mock_adapter_round_trip_healthy_world,
    test_mock_adapter_round_trip_fault_breaks_constraint,
    test_mock_adapter_round_trip_intervene_restores_constraint,
    test_mock_adapter_render_observation_returns_pydantic_model,
    test_mock_adapter_diagnose_returns_string,
    test_partial_subclass_still_abstract,
]


def _run_all_as_script():
    """Run all tests and print PASS/FAIL for each. Returns 0 if all pass, 1 otherwise."""
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

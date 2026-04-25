"""
CodeRepairAdapter (hardcoded V1)
================================

This is a CONCRETE DOMAIN ADAPTER for the "code repair" domain.
It plugs directly into Person B's CausalRepairEnv via the 7-method
BaseAdapter contract. No AST, no subprocess, no pytest needed.

HOW IT WORKS:
    The "world" is just a Python dict with four keys:
        - entities  : {function_name: source_code_string}
        - edges     : list of (caller, callee) pairs (call graph, empty for now)
        - tests     : {test_name: {status, msg, spec}}
        - meta      : {fault_ground_truth, domain}

    A "test spec" says: "call function X with args Y and expect result Z".
    propagate() actually runs those tests (using exec + call) and records pass/fail.

EPISODE FLOW (what the LLM experiences):
    1. generate_world()  -> builds a clean world; all 3 tests pass
    2. inject_fault()    -> silently changes add(x+y) to add(x-y); test_add now fails
    3. render_observation() -> LLM sees [VIOLATED] test_add but doesn't see why
    4. LLM calls diagnose("add") -> LLM sees the mutated source "return x - y"
    5. LLM calls intervene("add", "def add(x,y): return x+y") -> source replaced in world
    6. LLM calls propagate() -> tests re-run; all green
    7. LLM calls commit_repair(...) -> env calls check_constraints() -> True -> reward +1

UPGRADE PATH (Phase 2):
    - Replace _HEALTHY_SOURCES with ast.parse() of a real file
    - Replace _TEST_SPECS with discovery from test_calculator.py
    - Replace propagate()'s exec with subprocess.run(["pytest", ...])
    The 7-method contract stays identical across both versions.
"""
from __future__ import annotations

from typing import Any, Dict

from models import CausalrepairObservation
from server.base_adapter import BaseAdapter


# ---------------------------------------------------------------------------
# Module-level constants (the "hardcoded world")
# ---------------------------------------------------------------------------

# The source code of each function in its HEALTHY (correct) state.
# inject_fault() overwrites one of these with a buggy version.
_HEALTHY_SOURCES: Dict[str, str] = {
    "add": "def add(x, y):\n    return x + y\n",
    "sub": "def sub(x, y):\n    return x - y\n",
    "mul": "def mul(x, y):\n    return x * y\n",
}

# Test specifications: for each named test, which function to call,
# what arguments to pass, and what the correct return value is.
# propagate() uses this to run the tests without needing pytest.
_TEST_SPECS: Dict[str, Dict[str, Any]] = {
    "test_add": {"fn": "add", "args": (1, 2), "expected": 3},
    "test_sub": {"fn": "sub", "args": (5, 2), "expected": 3},
    "test_mul": {"fn": "mul", "args": (2, 3), "expected": 6},
}

# Human-readable description of each function's DOCUMENTED behaviour.
# This is what the LLM sees in the RULES section — it describes the
# intended contract, NOT the current (potentially buggy) source.
# After inject_fault(), the RULES still say "x + y" but the actual
# source says "x - y". That contradiction is the LLM's first clue.
_RULE_DESCRIPTIONS: Dict[str, str] = {
    "add": "add(x, y) -> x + y",
    "sub": "sub(x, y) -> x - y",
    "mul": "mul(x, y) -> x * y",
}


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class CodeRepairAdapter(BaseAdapter):
    """Hardcoded toy-code domain: 3 functions, 3 tests, 1 hardcoded fault."""

    # Tells Person B's env (and render_observation) which domain this is.
    domain_name = "code"

    # ------------------------------------------------------------------
    # Method 1: generate_world
    # ------------------------------------------------------------------
    def generate_world(self) -> Dict[str, Any]:
        """
        Build a fresh, fault-free world dict and return it.

        The world contains:
          - entities: copy of _HEALTHY_SOURCES (all correct function bodies)
          - edges:    empty list (call graph; not used in V1)
          - tests:    one entry per test, all starting as "unknown"
          - meta:     fault_ground_truth=None (no fault injected yet)

        propagate() is called at the end so the test statuses are "pass"
        from the moment the world is created (not "unknown").
        """
        world: Dict[str, Any] = {
            # Shallow copy so callers can't accidentally mutate _HEALTHY_SOURCES
            "entities": dict(_HEALTHY_SOURCES),

            # Call graph edges — empty now, Phase 2 will fill this from ast
            "edges": [],

            # One test entry per test spec. spec is copied so propagate()
            # can read it without re-reading the module-level constant.
            "tests": {
                name: {"status": "unknown", "msg": "", "spec": dict(spec)}
                for name, spec in _TEST_SPECS.items()
            },

            # meta holds anything that doesn't fit elsewhere.
            # fault_ground_truth stays None until inject_fault() is called.
            "meta": {
                "fault_ground_truth": None,
                "domain": "code",
            },
        }

        # Run all tests immediately so the world starts with accurate statuses.
        # After this call, all three tests should be "pass".
        self.propagate(world)
        return world

    # ------------------------------------------------------------------
    # Method 2: inject_fault
    # ------------------------------------------------------------------
    def inject_fault(self, world: Dict[str, Any]) -> None:
        """
        Silently break one function in the world (in-place, returns None).

        WHAT IT DOES:
            Replaces add(x, y): return x + y
            with    add(x, y): return x - y   <- bug: subtraction instead of addition

        The LLM is NOT told what changed. It only sees [VIOLATED] test_add
        in the observation and must diagnose the root cause.

        fault_ground_truth records the real answer (used by the env for reward
        evaluation, hidden from the LLM).
        """
        # Overwrite the "add" source in-place — the world dict is mutated directly
        world["entities"]["add"] = "def add(x, y):\n    return x - y\n"

        # Record the true fault so the env can later verify the LLM's rationale
        world["meta"]["fault_ground_truth"] = {
            "entity": "add",         # which function was broken
            "mutation": "add_to_sub",# what the mutation was
            "broken_test": "test_add",# which test it breaks
        }

        # Re-run all tests so statuses reflect the new broken source immediately.
        # After this, test_add -> "fail", test_sub/mul -> "pass".
        self.propagate(world)

    # ------------------------------------------------------------------
    # Method 3: render_observation
    # ------------------------------------------------------------------
    def render_observation(self, world: Dict[str, Any]) -> CausalrepairObservation:
        """
        Build the 5-section text the LLM reads, packed into a CausalrepairObservation.

        The five sections are non-negotiable (same order, same headers, every domain):
            DOMAIN, STATE, RULES, CONSTRAINTS, AVAILABLE ACTIONS

        description (str) — the text the LLM reads.
        extra (dict)      — structured data for logging/debugging (not shown to LLM).

        Note: RULES shows the DOCUMENTED behaviour (from _RULE_DESCRIPTIONS),
        not the current source. STATE shows PASSING/FAILING derived from test results.
        """
        # STATE section: one line per function, showing PASSING/FAILING/UNKNOWN.
        # Derived from which tests are currently failing for that function.
        state_lines = "\n".join(
            f"  {fn}: {self._fn_status(fn, world)}" for fn in world["entities"]
        )

        # RULES section: one line per function, showing its documented contract.
        # This is intentionally the "spec" not the "implementation" —
        # after inject_fault, rules still say "x + y" but the code says "x - y".
        rules_lines = "\n".join(
            f"  {fn}: {_RULE_DESCRIPTIONS.get(fn, '<no description>')}"
            for fn in world["entities"]
        )

        # CONSTRAINTS section: one line per test, showing [OK] or [VIOLATED].
        # Derived from world["tests"][name]["status"].
        constraint_lines = "\n".join(
            f"  [{self._constraint_tag(info['status'])}] {test_name}"
            for test_name, info in world["tests"].items()
        )

        # Assemble the full observation text. Format is locked across all adapters.
        description = (
            f"DOMAIN: {self.domain_name}\n"
            f"STATE:\n{state_lines}\n"
            f"RULES:\n{rules_lines}\n"
            f"CONSTRAINTS:\n{constraint_lines}\n"
            f"AVAILABLE ACTIONS:\n"
            f'  diagnose("entity_name")\n'
            f'  intervene("entity_name", "value")\n'
            f"  propagate()\n"
            f'  commit_repair("entity_name", "value", rationale="...")'
        )

        # extra is a structured summary for logging — Person B's env passes
        # this through info dict; the LLM never sees it directly.
        extra = {
            "entities": list(world["entities"].keys()),
            "tests": {t: info["status"] for t, info in world["tests"].items()},
            "fault_ground_truth": world["meta"].get("fault_ground_truth"),
        }
        return CausalrepairObservation(description=description, extra=extra)

    # ------------------------------------------------------------------
    # Method 4: diagnose
    # ------------------------------------------------------------------
    def diagnose(self, world: Dict[str, Any], entity: str) -> str:
        """
        Return a human-readable diagnostic report for one entity (read-only).

        The LLM calls this when it wants to inspect a specific function.
        The report reveals:
          - which tests are related to this function
          - which of those tests are currently failing (and why)
          - the actual current source code of the function

        This is the key "reveal" moment: after inject_fault(), the source
        shows "return x - y" while the rules say "x + y". The LLM can now
        form a hypothesis about the root cause.
        """
        # Guard: if the LLM names an entity that doesn't exist, don't crash
        if entity not in world["entities"]:
            return f"[DIAGNOSE] unknown entity: {entity}"

        src = world["entities"][entity]

        # Find all tests that exercise this specific function
        related_tests = [
            t for t, info in world["tests"].items() if info["spec"]["fn"] == entity
        ]

        # Narrow to just the ones that are currently failing
        failing = [t for t in related_tests if world["tests"][t]["status"] == "fail"]

        # Build the failing-tests section; if none are failing, say so
        failing_lines = "\n".join(
            f"    {t}: {world['tests'][t]['msg']}" for t in failing
        ) or "    (none)"

        return (
            f"[DIAGNOSE] {entity}\n"
            f"  related_tests={related_tests}\n"
            f"  failing_tests:\n{failing_lines}\n"
            f"  src=\n{src}"  # The actual current source — may contain the bug
        )

    # ------------------------------------------------------------------
    # Method 5: intervene
    # ------------------------------------------------------------------
    def intervene(self, world: Dict[str, Any], entity: str, value: Any) -> None:
        """
        Replace a function's source code in the world (in-place, returns None).

        The LLM calls this when it thinks it knows what the fix is.
        value is expected to be the new function source string.

        After intervene(), tests are marked "unknown" because we haven't re-run
        them yet. The LLM must call propagate() to find out if the fix worked,
        then commit_repair() to finalise.

        Example:
            intervene(world, "add", "def add(x, y):\\n    return x + y\\n")
        """
        # Silently ignore interventions on entities that don't exist
        if entity not in world["entities"]:
            return

        # Accept any value; coerce non-strings to string (safety net)
        world["entities"][entity] = value if isinstance(value, str) else str(value)

        # Mark all test results as stale — they were computed against the old
        # source and are no longer reliable. propagate() will refresh them.
        for info in world["tests"].values():
            info["status"] = "unknown"
            info["msg"] = ""

    # ------------------------------------------------------------------
    # Method 6: propagate
    # ------------------------------------------------------------------
    def propagate(self, world: Dict[str, Any]) -> None:
        """
        Re-run all tests against the current entity sources (in-place, returns None).

        HOW IT WORKS (no subprocess, no pytest):
            For each test spec {"fn": "add", "args": (1,2), "expected": 3}:
              1. Grab the current source of function "add" from world["entities"]
              2. exec() that source into a sandbox dict — this defines the function
                 in an isolated namespace (no risk of polluting global state)
              3. Call sandbox["add"](1, 2) and compare the result to 3
              4. Write "pass" or "fail" (+ error message) back into world["tests"]

        After propagate(), world["tests"] accurately reflects the current state
        of world["entities"]. Any test whose function was mutated by inject_fault()
        or intervene() will show its new status.

        Error handling: syntax errors, runtime exceptions, and wrong-type returns
        are all caught and recorded as "fail" with a descriptive message.
        The other tests are not affected — each runs in its own sandbox.
        """
        for test_name, info in world["tests"].items():
            spec = info["spec"]
            fn_name = spec["fn"]

            # Look up the current source for this function
            src = world["entities"].get(fn_name)
            if src is None:
                # Should not happen with hardcoded world, but be safe
                info["status"] = "fail"
                info["msg"] = f"missing entity: {fn_name}"
                continue

            # Isolated sandbox: exec() puts the function definition into this dict.
            # Using a fresh dict per test means tests can't pollute each other.
            sandbox: Dict[str, Any] = {}
            try:
                exec(src, sandbox)          # defines fn_name inside sandbox

                fn = sandbox.get(fn_name)
                if not callable(fn):
                    info["status"] = "fail"
                    info["msg"] = f"{fn_name} not callable after exec"
                    continue

                actual = fn(*spec["args"])  # call add(1, 2) or sub(5, 2) etc.

                if actual == spec["expected"]:
                    info["status"] = "pass"
                    info["msg"] = ""
                else:
                    info["status"] = "fail"
                    info["msg"] = f"expected {spec['expected']!r}, got {actual!r}"

            except Exception as exc:
                # Covers SyntaxError (bad source), NameError, TypeError, etc.
                info["status"] = "fail"
                info["msg"] = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------
    # Method 7: check_constraints
    # ------------------------------------------------------------------
    def check_constraints(self, world: Dict[str, Any]) -> bool:
        """
        Return True only if EVERY test in world["tests"] has status "pass".

        Person B's env calls this when the LLM issues commit_repair().
        If True  -> the repair worked -> env awards positive reward and ends episode.
        If False -> the repair didn't fix everything -> env awards -0.5 and continues.

        Important: this reads the stored test statuses, not re-running the tests.
        The LLM must have called propagate() first to get up-to-date statuses.
        """
        return all(info["status"] == "pass" for info in world["tests"].values())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fn_status(fn: str, world: Dict[str, Any]) -> str:
        """
        Derive a function's display status from its related test results.

        Maps: any failing test -> FAILING
              all passing     -> PASSING
              mix of unknown  -> UNKNOWN
              no tests at all -> UNTESTED
        """
        related = [
            info["status"]
            for info in world["tests"].values()
            if info["spec"]["fn"] == fn  # only tests that exercise this function
        ]
        if not related:
            return "UNTESTED"
        if any(s == "fail" for s in related):
            return "FAILING"
        if all(s == "pass" for s in related):
            return "PASSING"
        return "UNKNOWN"

    @staticmethod
    def _constraint_tag(status: str) -> str:
        """Convert a test status string to the [OK] / [VIOLATED] tag shown in observations."""
        return {"pass": "OK", "fail": "VIOLATED"}.get(status, "UNKNOWN")

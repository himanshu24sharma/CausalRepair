"""
verify.py — Interactive step-by-step verification of CodeRepairAdapter.

Run this from the package root (CausalRepair/CausalRepair/):
    ..\\venv\\Scripts\\python.exe verify.py

This script calls every adapter method one at a time, prints the result,
and pauses for you to press Enter before moving to the next step.
Use it to visually confirm that each method works as expected.

No pytest, no test assertions — just direct function calls + print output.
"""
import sys
from pathlib import Path

# Make imports work when run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from server.code_repair_adapter import CodeRepairAdapter

DIVIDER = "=" * 60


def pause(label: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {label}")
    print(DIVIDER)
    input("  Press Enter to run this step...")
    print()


def show(title: str, value) -> None:
    print(f"  [{title}]")
    for line in str(value).splitlines():
        print(f"    {line}")
    print()


# ---------------------------------------------------------------------------
adapter = CodeRepairAdapter()
print("\nCodeRepairAdapter loaded. domain_name =", adapter.domain_name)


# STEP 1: generate_world
# ---------------------------------------------------------------------------
pause("STEP 1: generate_world()")
print("  Calling: adapter.generate_world()")
world = adapter.generate_world()
show("entities (function names)", list(world["entities"].keys()))
show("test statuses after generate_world", {
    t: info["status"] for t, info in world["tests"].items()
})
show("fault_ground_truth (should be None)", world["meta"]["fault_ground_truth"])
print("  All tests pass?", adapter.check_constraints(world))


# STEP 2: render_observation (healthy world)
# ---------------------------------------------------------------------------
pause("STEP 2: render_observation() — healthy world")
print("  Calling: adapter.render_observation(world)")
obs = adapter.render_observation(world)
print("  Type returned:", type(obs).__name__)
print()
print("  obs.description:")
for line in obs.description.splitlines():
    print("    " + line)
print()
print("  obs.extra (structured summary for logging):", obs.extra)


# STEP 3: inject_fault
# ---------------------------------------------------------------------------
pause("STEP 3: inject_fault(world) — introduce a silent bug")
print("  Calling: adapter.inject_fault(world)")
print("  Before: entities['add'] =", repr(world["entities"]["add"]))
adapter.inject_fault(world)
print("  After:  entities['add'] =", repr(world["entities"]["add"]))
print()
show("fault_ground_truth (the hidden answer)", world["meta"]["fault_ground_truth"])
show("test statuses after inject_fault", {
    t: info["status"] for t, info in world["tests"].items()
})
print("  All tests pass?", adapter.check_constraints(world))


# STEP 4: render_observation (broken world)
# ---------------------------------------------------------------------------
pause("STEP 4: render_observation() — broken world (what the LLM sees)")
print("  Calling: adapter.render_observation(world)")
obs = adapter.render_observation(world)
print("  obs.description (this is what the LLM reads):")
print()
for line in obs.description.splitlines():
    print("    " + line)
print()
print("  Notice: RULES still says 'x + y' but STATE shows 'add: FAILING'")
print("  The LLM sees a violated constraint but doesn't know WHY yet.")


# STEP 5: diagnose
# ---------------------------------------------------------------------------
pause("STEP 5: diagnose(world, 'add') — LLM asks 'what's wrong with add?'")
print('  Calling: adapter.diagnose(world, "add")')
diag = adapter.diagnose(world, "add")
print()
print("  Diagnose output:")
for line in diag.splitlines():
    print("    " + line)
print()
print("  Now the LLM can see the actual source: 'return x - y'")
print("  The rules say 'x + y'. That's the bug.")
print()
print("  Also try diagnosing an entity that doesn't exist:")
print('  Calling: adapter.diagnose(world, "nonexistent")')
print(" ", adapter.diagnose(world, "nonexistent"))


# STEP 6: intervene
# ---------------------------------------------------------------------------
pause("STEP 6: intervene(world, 'add', <fixed source>) — LLM applies a fix")
fixed_source = "def add(x, y):\n    return x + y\n"
print("  Calling: adapter.intervene(world, 'add', fixed_source)")
print("  Fixed source:", repr(fixed_source))
adapter.intervene(world, "add", fixed_source)
print()
print("  After intervene:")
print("    entities['add'] =", repr(world["entities"]["add"]))
print()
show("test statuses after intervene (stale until propagate)", {
    t: info["status"] for t, info in world["tests"].items()
})
print("  Note: all tests are 'unknown' — the fix hasn't been verified yet.")


# STEP 7: propagate
# ---------------------------------------------------------------------------
pause("STEP 7: propagate(world) — re-run all tests to verify the fix")
print("  Calling: adapter.propagate(world)")
adapter.propagate(world)
show("test statuses after propagate", {
    t: info["status"] for t, info in world["tests"].items()
})
print("  All tests pass?", adapter.check_constraints(world))
print("  check_constraints returns True -> commit_repair will earn reward +1.0")


# STEP 8: render_observation (repaired world)
# ---------------------------------------------------------------------------
pause("STEP 8: render_observation() — repaired world")
obs = adapter.render_observation(world)
print("  obs.description (what LLM sees after repair):")
print()
for line in obs.description.splitlines():
    print("    " + line)
print()
print("  All constraints show [OK]. The LLM is ready to call commit_repair.")


# STEP 9: what happens with wrong intervention
# ---------------------------------------------------------------------------
pause("BONUS STEP: what if the LLM makes a wrong intervention?")
print("  Start fresh...")
world2 = adapter.generate_world()
adapter.inject_fault(world2)
print("  inject_fault done. Now LLM makes a wrong guess:")
wrong_source = "def add(x, y):\n    return x * y\n"
print("  intervene('add', 'return x * y')  <- wrong fix")
adapter.intervene(world2, "add", wrong_source)
adapter.propagate(world2)
show("test statuses after wrong fix", {
    t: info["status"] for t, info in world2["tests"].items()
})
print("  check_constraints:", adapter.check_constraints(world2))
print("  -> False. If LLM calls commit_repair now, env will return reward = -0.5")
print("     The LLM must try again.")


print()
print(DIVIDER)
print("  Verification complete.")
print("  All 7 adapter methods demonstrated.")
print(DIVIDER)
print()

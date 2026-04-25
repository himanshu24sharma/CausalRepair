"""
repl.py — Interactive REPL for the CausalRepair environment.

Lets you drive a CausalrepairEnvironment by hand: you type actions, the env
runs them, and you see the resulting observation, reward, and done flag —
exactly what an LLM/agent would see over the wire.

This is the human-driver complement to:
    - verify.py : steps through the adapter's 7 methods directly
    - app.py    : exposes the env over HTTP/WebSocket for real agents

Run from the package root (CausalRepair/CausalRepair/):
    ..\\venv\\Scripts\\python.exe repl.py
    ..\\venv\\Scripts\\python.exe repl.py --adapter hydraulic
    ..\\venv\\Scripts\\python.exe repl.py --adapter code --max-steps 30

Commands
--------
    diagnose <entity>
    intervene <entity> <value>          value = rest of the line
    intervene <entity> <<END            heredoc; finish with a line "END"
    propagate
    commit <entity> [rationale...]      alias: commit_repair
    obs                                  re-print current observation (no step)
    reset                                start a new episode
    help | ?                             show this message
    quit | exit | q                      leave the REPL (also: EOF / Ctrl-C)
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

# Make `from models import ...` and `from server.* import ...` resolve when
# this file is launched from any working directory (mirrors verify.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import CausalrepairAction
from server.CausalRepair_environment import CausalrepairEnvironment
from server.code_repair_adapter import CodeRepairAdapter
from server.mock_adapter import MockAdapter


ADAPTERS = {
    "code": CodeRepairAdapter,
    "hydraulic": MockAdapter,
}

PROMPT = "(causalrepair) "

HELP = """\
Commands:
  diagnose <entity>                 Ask the env about one entity.
  intervene <entity> <value>        Apply a single-line change.
  intervene <entity> <<END          Multi-line value; end with a line "END".
  propagate                         Re-run the world's rules / tests.
  commit <entity> [rationale...]    Finish the episode (alias: commit_repair).
  obs                               Re-print the current observation.
  reset                             Start a new episode.
  help | ?                          Show this message.
  quit | exit | q                   Leave the REPL.
"""


def _print_observation(description: str) -> None:
    for line in description.splitlines() or [""]:
        print(f"  {line}")


def _print_step(label: str, result) -> None:
    """Pretty-print a StepResult-like object."""
    obs = result.observation
    description = (
        obs.description if hasattr(obs, "description")
        else obs.get("description", "") if isinstance(obs, dict)
        else str(obs)
    )
    print(f"\n--- {label} ---")
    _print_observation(description)
    print(f"  reward = {result.reward}   done = {result.done}   info = {result.info}")


def _read_heredoc(end_marker: str) -> str:
    """Read lines from stdin until one equals `end_marker`; return joined text."""
    lines: list[str] = []
    while True:
        try:
            line = input("... ")
        except EOFError:
            break
        if line == end_marker:
            break
        lines.append(line)
    return ("\n".join(lines) + "\n") if lines else ""


def _build_action(cmd: str, rest: str):
    """
    Translate a parsed command line into a CausalrepairAction.

    Returns the action on success, or None if the command was malformed
    (in which case a usage hint has already been printed).
    """
    if cmd == "diagnose":
        if not rest:
            print("usage: diagnose <entity>")
            return None
        return CausalrepairAction(action_type="diagnose", target=rest)

    if cmd == "intervene":
        target, _, value = rest.partition(" ")
        if not target or not value:
            print("usage: intervene <entity> <value>   (value '<<END' opens a heredoc)")
            return None
        if value.startswith("<<"):
            marker = value[2:].strip() or "END"
            value = _read_heredoc(marker)
        return CausalrepairAction(action_type="intervene", target=target, value=value)

    if cmd == "propagate":
        return CausalrepairAction(action_type="propagate")

    if cmd in ("commit", "commit_repair"):
        if not rest:
            print("usage: commit <entity> [rationale...]")
            return None
        target, _, rationale = rest.partition(" ")
        return CausalrepairAction(
            action_type="commit_repair",
            target=target,
            rationale=rationale or None,
        )

    return None  # unreachable; caller handles unknown verbs first


def _handle(env, line: str) -> bool:
    """Process one input line. Return False to quit, True to continue."""
    raw = line.strip()
    if not raw:
        return True

    cmd, _, rest = raw.partition(" ")
    cmd = cmd.lower()
    rest = rest.strip()

    if cmd in ("quit", "exit", "q"):
        return False
    if cmd in ("help", "?"):
        print(HELP)
        return True
    if cmd == "obs":
        obs = env.adapter.render_observation(env.world)
        print()
        _print_observation(obs.description)
        return True
    if cmd == "reset":
        result = env.reset()
        _print_step("RESET", result)
        return True

    if cmd not in ("diagnose", "intervene", "propagate", "commit", "commit_repair"):
        print(f"unknown command: {cmd!r}. Type 'help'.")
        return True

    action = _build_action(cmd, rest)
    if action is None:
        return True

    try:
        result = env.step(action)
    except Exception:  # REPL must survive adapter-side errors
        print("step raised an exception:")
        traceback.print_exc()
        return True

    _print_step(cmd.upper(), result)
    if result.done:
        print("\n[episode finished -- type 'reset' to play again, or 'quit']")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive REPL for the CausalRepair environment."
    )
    parser.add_argument(
        "--adapter",
        choices=sorted(ADAPTERS),
        default="code",
        help="Domain adapter to load (default: code).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Episode step budget (default: 20).",
    )
    args = parser.parse_args()

    adapter_cls = ADAPTERS[args.adapter]
    env = CausalrepairEnvironment(adapter=adapter_cls(), max_steps=args.max_steps)

    print(f"CausalRepair REPL -- adapter: {args.adapter} ({adapter_cls.__name__})")
    print("Type 'help' for commands, 'quit' to leave.\n")
    print("Initial observation:")
    _print_observation(env.adapter.render_observation(env.world).description)
    print()

    while True:
        try:
            line = input(PROMPT)
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not _handle(env, line):
            break

    print("Goodbye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

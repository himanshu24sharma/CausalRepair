"""
Inference Script for CausalRepair Environment
============================================
This script runs one or more episodes against the in-process environment and
uses an OpenAI-compatible API for action selection.

CLI usage:
    python inference.py                                  # 1 episode, verbose logs
    python inference.py --episodes 10                    # 10 episodes, verbose logs
    python inference.py --episodes 10 --json             # 10 episodes, JSON-lines
    python inference.py --episodes 10 --json --max-steps 12

In ``--json`` mode the script prints exactly one JSON object per line, one per
episode, of the form ``{"reward": <float>, "steps": <int>, "success": <bool>}``.
All other stdout (env debug prints, [START]/[STEP]/[END]) is suppressed in
that mode so the output is cleanly parseable as JSON-lines.
"""

import argparse
import contextlib
import io
import json
import os
import sys

from models import CausalrepairAction
from openai import OpenAI
from server.CausalRepair_environment import CausalrepairEnvironment
from server.code_repair_adapter import CodeRepairAdapter

LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv(
    "API_BASE_URL", "https://router.huggingface.co/v1"
)
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-9B:together")
BENCHMARK = "CausalRepair-v0"


# ---------------------------------------------------------------------------
# Module-scope SYSTEM_PROMPT (hoisted from main() so the GRPO notebook can
# `from inference import SYSTEM_PROMPT` and reuse it verbatim).
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a CausalRepair agent. Your goal is to make the system satisfy all "
    "constraints with as few diagnose calls as possible, and to maximize total reward. "
    "Given the observation, reply ONLY with a valid action as a JSON object. "
    "Do NOT use markdown, code blocks, or any extra text—just the JSON object. "
    "The JSON must use these keys: 'action_type', 'target', 'value', 'rationale', 'payload'. "
    "Available actions are: "
    "diagnose(\"entity\"): Inspect a function to understand why it is failing. "
    "intervene(\"entity\", \"value\"): Propose a fix for a function by providing new code. "
    "propagate(): Re-run all tests after an intervention. "
    "commit_repair(\"entity\", \"value\", rationale=\"...\"): Submit your final repair and end the episode. "
    "**Important:** an intervene() does NOT re-run the tests; it only swaps the source "
    "code and marks every test status as 'unknown'. After you call intervene, you MUST "
    "call propagate() as your very next action to observe whether your fix resolved the "
    "violated tests. Only after propagate shows the test statuses (all [OK], or [VIOLATED] "
    "if you intend to try another fix) should you consider committing. "
    "**If all tests are PASSING and all constraints are [OK], you MUST call commit_repair(...) "
    "as your next action. After committing, do not intervene again.** "
    "You must NOT use commit_repair as your first action when tests are still failing, "
    "and you must NOT call commit_repair while any test status is 'unknown'. "
    "Always diagnose the failing entity first. After diagnosing, if you have enough information, "
    "proceed to intervene on the failing entity, then propagate, and only then use commit_repair "
    "to finish. Do not repeat diagnose unless new information is needed. "
    "Example: {\"action_type\": \"diagnose\", \"target\": \"add\", \"value\": null, \"rationale\": null, \"payload\": {}}"
)


def log_start(env: str, model: str) -> None:
    print(f"[START] env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def build_prompt(obs):
    # Simple stringification for now
    return str(obs)


def _world_healthy(obs) -> bool:
    """Return True iff every test in the observation has status 'pass'.

    Accepts either a CausalrepairObservation pydantic instance or a plain
    obs dict, and tolerates both flat ({test: 'pass'}) and nested
    ({test: {'status': 'pass'}}) test maps.
    """
    if obs is None:
        return False
    if hasattr(obs, "model_dump"):
        obs_dict = obs.model_dump()
    elif isinstance(obs, dict):
        obs_dict = obs
    else:
        return False
    extra = obs_dict.get("extra") or {}
    tests = extra.get("tests") or {}
    if not tests:
        return False
    for v in tests.values():
        status = v.get("status") if isinstance(v, dict) else v
        if status != "pass":
            return False
    return True


def compute_reward(
    action: CausalrepairAction,
    done: bool,
    info: dict,
    max_steps: int,
    diagnose_budget: int,
    was_healthy_before: bool = False,
) -> float:
    """Commit-centric shaped reward; the env itself stays reward-free.

    Strategy:
      - diagnose:               -0.05 (small per-call penalty)
      - intervene:               0.0  (main reward is on commit)
      - propagate:               0.0  base, plus a shaping bonus of
                                 +0.05 * newly_known when a propagate call
                                 turns one or more tests from "unknown"
                                 into a known status ("pass" / "fail").
                                 This teaches the LLM that calling
                                 propagate() after an intervene is what
                                 reveals whether the fix worked.
      - commit_repair:
          * constraints broken: -1.0
          * constraints ok:     +1.0 + efficiency_bonus + budget_bonus
              efficiency_bonus = 0.3 * max(0, 1 - steps / max_steps)
              budget_bonus     = 0.2 if diagnose_calls <= diagnose_budget else 0.0
      - episode ended without commit (timeout / unknown action): -0.5
      - extra: -0.1 per step taken AFTER the world was already healthy
        (only applied to non-commit actions; encourages the agent to
        commit_repair as soon as all tests pass instead of looping).
    """
    action_type = action.action_type
    constraints_ok = bool(info.get("constraints_ok", False))
    steps = int(info.get("steps", 0))
    diagnose_calls = int(info.get("diagnose_calls", 0))
    newly_known = int(info.get("newly_known", 0))

    # Propagate shaping bonus: +0.05 per test that flipped from "unknown" to
    # a known status during this step. Computed up-front so it can be added
    # to the action's reward consistently in every branch below (including
    # timeouts), which keeps the credit assignment for "calling propagate"
    # independent of whether the episode happens to end on this step.
    propagate_shaping = 0.0
    if action_type == "propagate" and newly_known > 0:
        propagate_shaping = 0.05 * newly_known

    if action_type == "commit_repair":
        if not constraints_ok:
            return -1.0
        success = 1.0
        efficiency = 0.3 * max(0.0, 1.0 - steps / max_steps)
        budget_bonus = 0.2 if diagnose_calls <= diagnose_budget else 0.0
        return success + efficiency + budget_bonus

    # Timeout: episode ended without a commit (covers any non-commit action).
    # We still pay out the propagate shaping bonus so the LLM gets credit for
    # the information-revealing action even if it ran out of budget.
    if done:
        return -0.5 + propagate_shaping

    if action_type == "diagnose":
        reward = -0.05
    elif action_type in ("intervene", "propagate"):
        reward = 0.0
    else:
        reward = 0.0

    reward += propagate_shaping

    # Discourage spinning after the world is already healthy.
    if was_healthy_before:
        reward -= 0.1

    return reward


# ---------------------------------------------------------------------------
# Episode runner. Returns (total_reward, steps, success) for a single
# episode. Verbose logging is unchanged from the original main() loop and is
# silenced upstream (via redirect_stdout) when running in --json mode.
# ---------------------------------------------------------------------------

def _run_one_episode(env, client, max_steps: int, verbose: bool) -> tuple:
    reset_result = env.reset()
    obs = reset_result.observation
    done = False
    total_reward = 0.0
    step_count = 0
    rewards: list = []

    if verbose:
        log_start(env="CausalRepair-v0", model=MODEL_NAME)

    last_intervened_entity = None
    last_intervened_value = None
    last_action_type = None
    last_constraints_ok = False

    while not done and step_count < max_steps:
        was_healthy_before = _world_healthy(obs)
        try:
            prompt = build_prompt(obs)
            completion = client.chat.completions.create(
                model="Qwen/Qwen3.5-9B",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=81920,
                temperature=1.0,
                top_p=0.95,
                presence_penalty=1.5,
                extra_body={
                    "top_k": 20,
                },
            )
            action_json = (completion.choices[0].message.content or "").strip()
            action = CausalrepairAction(**json.loads(action_json))
        except Exception:
            action = CausalrepairAction(action_type="commit_repair")

        if action.action_type == "intervene":
            last_intervened_entity = action.target
            last_intervened_value = action.value

        try:
            if verbose:
                print(action)
            step_result = env.step(action)
            if verbose:
                print("action done")
                print(
                    f"[DEBUG] Observation as dict: "
                    f"{getattr(step_result.observation, 'model_dump', lambda: step_result.observation)()}",
                    flush=True,
                )
            obs = step_result.observation
            done = step_result.done
            info = step_result.info
            reward = compute_reward(
                action=action,
                done=done,
                info=info,
                max_steps=env.max_steps,
                diagnose_budget=env.diagnose_budget,
                was_healthy_before=was_healthy_before,
            )
            error = None
        except Exception as e:
            error = str(e)
            obs, reward, done, info = None, 0.0, True, {}

        if verbose:
            log_step(
                step=step_count + 1,
                action=str(action),
                reward=reward,
                done=done,
                error=error,
            )

        rewards.append(reward)
        total_reward += reward
        step_count += 1
        last_action_type = action.action_type
        last_constraints_ok = bool((info or {}).get("constraints_ok", False))

        # Force-commit if the world is already healthy (all tests passing and
        # all constraints satisfied). This overrides whatever the LLM intends
        # to do next, so it can't keep intervening after a successful repair.
        if (
            error is None
            and not done
            and _world_healthy(obs)
            and step_count < max_steps
        ):
            commit_action = CausalrepairAction(
                action_type="commit_repair",
                target=last_intervened_entity,
                value=last_intervened_value,
                rationale="All tests pass; fix is minimal and correct.",
            )
            try:
                if verbose:
                    print(commit_action)
                step_result = env.step(commit_action)
                if verbose:
                    print("action done (forced commit)")
                    print(
                        f"[DEBUG] Observation as dict: "
                        f"{getattr(step_result.observation, 'model_dump', lambda: step_result.observation)()}",
                        flush=True,
                    )
                obs = step_result.observation
                done = step_result.done
                info = step_result.info
                reward = compute_reward(
                    action=commit_action,
                    done=done,
                    info=info,
                    max_steps=env.max_steps,
                    diagnose_budget=env.diagnose_budget,
                    was_healthy_before=True,
                )
                error = None
            except Exception as e:
                error = str(e)
                obs, reward, done, info = None, 0.0, True, {}
            if verbose:
                log_step(
                    step=step_count + 1,
                    action=str(commit_action),
                    reward=reward,
                    done=done,
                    error=error,
                )
            rewards.append(reward)
            total_reward += reward
            step_count += 1
            last_action_type = commit_action.action_type
            last_constraints_ok = bool((info or {}).get("constraints_ok", False))
            break

    success = bool(last_action_type == "commit_repair" and last_constraints_ok)

    if verbose:
        log_end(success=success, steps=step_count, rewards=rewards)

    return total_reward, step_count, success


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run CausalRepair inference episodes.")
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    p.add_argument(
        "--json",
        dest="json_mode",
        action="store_true",
        help="Emit one JSON object per line, suppress all other stdout.",
    )
    p.add_argument("--max-steps", type=int, default=10, help="Max steps per episode.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    env = CausalrepairEnvironment(adapter=CodeRepairAdapter())
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    client = OpenAI(base_url=LLM_BASE_URL, api_key=api_key)

    real_stdout = sys.stdout

    for _ in range(args.episodes):
        if args.json_mode:
            # Silence everything (including env debug prints) during the
            # episode so the JSON-lines output stays parseable. We restore
            # stdout afterwards and print exactly one JSON object.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                total_reward, step_count, success = _run_one_episode(
                    env=env, client=client, max_steps=args.max_steps, verbose=False
                )
            real_stdout.write(
                json.dumps(
                    {
                        "reward": round(float(total_reward), 4),
                        "steps": int(step_count),
                        "success": bool(success),
                    }
                )
                + "\n"
            )
            real_stdout.flush()
        else:
            _run_one_episode(
                env=env, client=client, max_steps=args.max_steps, verbose=True
            )


if __name__ == "__main__":
    main()

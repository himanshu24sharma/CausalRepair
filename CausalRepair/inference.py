"""
Inference Script for CausalRepair Environment
============================================
This script runs a single episode against the in-process environment and uses an OpenAI-compatible API for action selection.
"""

import os
import json
from models import CausalrepairAction
from openai import OpenAI
from server.CausalRepair_environment import CausalrepairEnvironment
from server.mock_adapter import MockAdapter

LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "DummyModel")
BENCHMARK = "CausalRepair-v0"

def log_start(env: str, model: str) -> None:
    print(f"[START] env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)



def build_prompt(obs):
    # Simple stringification for now
    return str(obs)

def main():
    env = CausalrepairEnvironment(adapter=MockAdapter())
    API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    client = OpenAI(base_url=LLM_BASE_URL, api_key=API_KEY)
    reset_result = env.reset()
    obs = reset_result.observation
    done = False
    total_reward = 0
    step_count = 0
    max_steps = 20
    rewards = []
        
    log_start(env="CausalRepair-v0", model=MODEL_NAME)
    SYSTEM_PROMPT = "You are a CausalRepair agent. Given the observation, reply with a valid action as a JSON object."
    while not done and step_count < max_steps:
        try:
            prompt = build_prompt(obs)
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=64,
                stream=False,
            )
            action_json = (completion.choices[0].message.content or "").strip()
            action = CausalrepairAction(**json.loads(action_json))
        except Exception:
            action = CausalrepairAction(action_type="commit_repair")
        try:
            step_result = env.step(action)
            obs = step_result.observation
            reward = step_result.reward
            done = step_result.done
            info = step_result.info
            error = None
        except Exception as e:
            error = str(e)
            obs, reward, done, info = None, 0.0, True, {}
        log_step(step=step_count+1, action=str(action), reward=reward, done=done, error=error)
        rewards.append(reward)
        total_reward += reward
        step_count += 1
    log_end(success=done, steps=step_count, rewards=rewards)

if __name__ == "__main__":
    main()


# CausalRepair: Architecture, Design, and What We Built

## Overview

CausalRepair is an OpenEnv-compatible reinforcement learning environment for diagnosis and repair tasks. The main idea is to let an agent interact with a broken system the same way an engineer would: inspect the current state, diagnose the likely fault, apply an intervention, propagate the consequences, and then commit a repair when the constraints are satisfied.

In this round, we built a complete vertical slice for that idea:

- a reusable environment core with a stable action and observation interface,
- a pluggable adapter architecture so one environment can host multiple domains,
- a concrete code-repair domain with fault injection and executable test feedback,
- a FastAPI and OpenEnv server wrapper for interactive use,
- an agent loop for LLM-driven inference,
- local tooling for manual debugging and verification,
- and a test suite that checks the core behavior end to end.

The result is not just a demo script. It is a small platform for causal diagnosis tasks where the environment logic stays domain-agnostic and the domain-specific reasoning is pushed into adapters.

## The Problem We Wanted to Solve

Most repair benchmarks collapse directly to input-output prediction: the model sees a bug, emits a fix, and gets graded on the final answer. That setup skips the reasoning process that makes repair interesting.

We wanted a task structure where the agent has to:

1. observe symptoms,
2. decide what to inspect,
3. collect additional evidence,
4. propose a targeted intervention,
5. propagate the updated state,
6. and only then commit a repair.

That is why the environment exposes explicit causal actions instead of a single "submit answer" step.

## System Architecture

At a high level, the system is composed of five layers:

1. **Schema layer**: shared Pydantic models define the action and observation contract.
2. **Environment core**: the RL environment owns episode state, step counting, budgets, and termination logic.
3. **Adapter layer**: domain-specific implementations generate worlds, inject faults, diagnose entities, propagate consequences, and validate constraints.
4. **Serving layer**: FastAPI plus OpenEnv exposes the environment over HTTP and WebSocket.
5. **Agent and tooling layer**: inference, REPL, verification scripts, notebook work, and tests drive the environment.

The architecture is intentionally split so that the environment loop does not care whether the domain is code repair, hydraulic systems, configuration debugging, or something else. The environment only asks the adapter for world generation, mutation, observation rendering, and constraint evaluation.

```text
+-----------------------+
| LLM or Scripted Agent |
+-----------------------+
		|
		v
+---------------------+
| CausalrepairAction  |
+---------------------+
		|
		v
+---------------------------+
| CausalrepairEnvironment   |
+---------------------------+
	|                |
	|                +------------------------------+
	v                                               |
+-----------------------+                             |
| BaseAdapter Contract  |                             |
+-----------------------+                             |
	|                |                              |
	v                v                              v
+----------------+  +-------------+   +-----------------------------+
| CodeRepair     |  | MockAdapter |   | OpenEnv and FastAPI Server  |
| Adapter        |  | hydraulic   |   +-----------------------------+
+----------------+  +-------------+
	|
	v
+-----------------------------+
| StepResult and Observation  |
+-----------------------------+
		|
		v
+-----------------------+
| Returned to the Agent |
+-----------------------+
```

## End-to-End Flow

The runtime loop looks like this:

```text
LLM / scripted agent
	-> emits CausalrepairAction
	-> environment step(action)
	-> adapter mutates or inspects world
	-> adapter renders observation
	-> environment returns StepResult
	-> reward is computed by the inference loop
```

The episode starts with `reset()`:

- the adapter generates a healthy world,
- a hidden fault is injected,
- the initial observation is rendered,
- the agent sees symptoms but not the ground truth.

During `step()` the agent may choose one of four actions:

- `diagnose`: inspect a specific entity and retrieve a focused explanation,
- `intervene`: apply a proposed change,
- `propagate`: re-run the rules or tests so downstream state reflects the intervention,
- `commit_repair`: finalize the repair and end the episode.

This structure matters because it forces the model to act in a causal loop instead of treating the environment as a one-shot text generation task.

```text
LLM Agent                  Environment                  Adapter
---------                  -----------                  -------
	|                           |                          |
	| reset()                   |                          |
	|-------------------------->|                          |
	|                           | generate_world()         |
	|                           |------------------------->|
	|                           | inject_fault(world)      |
	|                           |------------------------->|
	|                           | render_observation()     |
	|                           |------------------------->|
	| initial observation       |                          |
	|<--------------------------|                          |
	|                           |                          |
	| step(diagnose)            |                          |
	|-------------------------->| diagnose(world, target)  |
	|                           |------------------------->|
	| diagnose_result           |                          |
	|<--------------------------|                          |
	|                           |                          |
	| step(intervene)           |                          |
	|-------------------------->| intervene(target, value) |
	|                           |------------------------->|
	|                           |                          |
	| step(propagate)           |                          |
	|-------------------------->| propagate(world)         |
	|                           |------------------------->|
	|                           | render_observation()     |
	|                           |------------------------->|
	| updated observation       |                          |
	|<--------------------------|                          |
	|                           |                          |
	| step(commit_repair)       |                          |
	|-------------------------->| check_constraints()      |
	|                           |------------------------->|
	| done + final info         |                          |
	|<--------------------------|                          |
```

## Core Environment Design

The heart of the project is the environment class in [round2/CausalRepair/server/CausalRepair_environment.py](round2/CausalRepair/server/CausalRepair_environment.py). It owns the episode-level mechanics:

- current world state,
- step count,
- diagnose budget,
- termination condition,
- last diagnosis result,
- and the observation returned to the agent.

The environment does **not** encode domain knowledge itself. Instead, it delegates the following responsibilities to the adapter:

- `generate_world()`
- `inject_fault()`
- `render_observation()`
- `diagnose()`
- `intervene()`
- `propagate()`
- `check_constraints()`

That separation is the main architectural decision in the project. It lets the environment stay small and generic while adapters define what a "world", a "fault", and a "repair" mean in each domain.

The environment also exposes a `state` property that returns the raw world, counters, done flag, and the latest observation payload. That makes debugging and inspection easier without changing the interaction contract used by the agent.

## Adapter Contract

The adapter contract is defined in [round2/CausalRepair/server/base_adapter.py](round2/CausalRepair/server/base_adapter.py). This file is effectively the interface boundary for the whole project.

We locked the adapter API around seven methods so that the environment core can remain unchanged while domains swap underneath it. This gives us two benefits:

- rapid experimentation with new causal domains,
- and stable integration with the environment, server, and agent loop.

The adapter also standardizes the observation format. Every domain is expected to render a description with the same five conceptual sections:

- `DOMAIN`
- `STATE`
- `RULES`
- `CONSTRAINTS`
- `AVAILABLE ACTIONS`

That formatting choice is subtle but important. It makes prompting simpler because the agent can rely on a consistent shape even when the underlying domain changes.

## CodeRepairAdapter: Our Main Domain

The main domain implementation lives in [round2/CausalRepair/server/code_repair_adapter.py](round2/CausalRepair/server/code_repair_adapter.py). This is the most concrete part of the system.

The adapter builds a miniature codebase inside the world state:

- three functions: `add`, `sub`, and `mul`,
- executable test specifications for each function,
- domain rules describing the intended behavior,
- metadata containing the hidden ground-truth fault.

The healthy world starts with correct source strings for all functions. Then `inject_fault()` silently mutates `add` from addition to subtraction. From the agent's perspective, it only sees that one constraint is violated. The actual buggy source is revealed later through `diagnose("add")`.

This gives us a simple but complete repair loop:

1. the world starts correct,
2. the environment injects a hidden bug,
3. tests fail,
4. the agent diagnoses the failing function,
5. the agent proposes a source-level intervention,
6. the adapter re-runs tests in `propagate()`,
7. the agent commits the repair once all constraints are satisfied.

The key design choice here is that `propagate()` executes the current function sources dynamically and evaluates test specs against them. That makes the environment stateful and causal: interventions have consequences, and those consequences are observable.

```text
[Healthy Sources]
	|
	v
[generate_world]
	|
	v
[inject_fault]
	|
	v
[Broken add function]
	|
	v
[test_add fails]
	|
	v
[diagnose add]
	|
	v
[buggy source revealed]
	|
	v
[intervene with repaired source]
	|
	v
[propagate reruns tests]
	|
	v
[all constraints satisfied]
	|
	v
[commit_repair]
```

## Reward Design

The environment itself returns neutral reward values and keeps the transition logic focused on state updates. Reward shaping is implemented in [round2/CausalRepair/inference.py](round2/CausalRepair/inference.py).

The reward strategy is commit-centric:

- a small penalty for `diagnose`,
- no direct reward for `intervene` or `propagate`,
- a positive reward for a correct `commit_repair`,
- an efficiency bonus for solving the task in fewer steps,
- a budget bonus for staying within the diagnose budget,
- and a negative outcome for committing or timing out without satisfying constraints.

This separation was deliberate. It keeps the environment reusable while making it easy to iterate on training incentives independently.

## Serving Layer and Deployment

The server entry point is [round2/CausalRepair/server/app.py](round2/CausalRepair/server/app.py). It wraps the environment with OpenEnv's `create_app(...)` helper and exposes it as a FastAPI application.

Two important ideas show up here:

- **adapter selection by environment variable**: `CR_ADAPTER` determines which domain adapter is loaded,
- **persistent environment instance**: the server builds one environment instance and serves it through the OpenEnv app.

Right now the registry contains:

- `code` -> `CodeRepairAdapter`
- `hydraulic` -> `MockAdapter`

That means the same serving layer can host different causal worlds without changing the outer API.

For packaging and deployment, we also added:

- [round2/CausalRepair/openenv.yaml](round2/CausalRepair/openenv.yaml) as the OpenEnv manifest,
- [round2/CausalRepair/pyproject.toml](round2/CausalRepair/pyproject.toml) for packaging and dependencies,
- the server requirements and Docker support under [round2/CausalRepair/server](round2/CausalRepair/server).

```text
[CR_ADAPTER env var]
		  |
		  v
[server/app.py registry]
	  /             \
	 /               \
	v                 v
[code]           [hydraulic]
	|                 |
	v                 v
[CodeRepairAdapter] [MockAdapter]
		  \         /
		   \       /
			v     v
	 [CausalrepairEnvironment]
				 |
				 v
		 [OpenEnv create_app]
				 |
				 v
		[HTTP and WebSocket API]
```

## Alternate Domain: Hydraulic Mock Adapter

To prove that the architecture is not tied to code repair, we added [round2/CausalRepair/server/mock_adapter.py](round2/CausalRepair/server/mock_adapter.py).

This adapter models a tiny hydraulic system with a valve, pressure, and alarm state. It uses the same environment contract but a completely different world representation and propagation rule.

That file matters because it validates the abstraction. Without a second adapter, the interface would only be theoretical. With it, we showed that the environment core really is reusable.

## Action and Observation Schema

The shared models live in [round2/CausalRepair/models.py](round2/CausalRepair/models.py). They define:

- `CausalrepairAction`
- `CausalrepairObservation`
- `StepResult`

The action schema supports four action types and includes optional fields for target, intervention value, rationale, and additional payload. The observation schema carries both the human-readable description and structured extra data.

This combination gives us a useful balance:

- the description is optimized for LLM reasoning,
- the structured fields are optimized for logging, debugging, and evaluation.

## Inference Loop

The agent runner is [round2/CausalRepair/inference.py](round2/CausalRepair/inference.py). It creates an in-process environment, builds prompts from observations, calls an OpenAI-compatible API, parses the returned JSON action, and steps the environment until termination.

This file is where the project turns from a static environment into an agent-evaluable benchmark. It demonstrates:

- model-driven action selection,
- prompt shaping around the environment state,
- reward computation,
- logging for each step,
- and automatic propagation after interventions.

Even though the benchmark is still small, this script establishes the exact loop we would later use for larger-scale evaluation.

## Human Debugging Tooling

We did not stop at the agent loop. We also built tools for humans to inspect and stress the environment:

- [round2/CausalRepair/verify.py](round2/CausalRepair/verify.py) steps through the adapter methods one by one and prints their outputs for manual verification.
- [round2/CausalRepair/repl.py](round2/CausalRepair/repl.py) provides an interactive command-line REPL so a person can play the role of the agent.
- [round2/CausalRepair/CausalRepair_OpenEnv_Training.ipynb](round2/CausalRepair/CausalRepair_OpenEnv_Training.ipynb) supports interactive experimentation in notebook form.

These tools are important for two reasons:

- they make debugging faster than working only through the HTTP surface,
- and they make the project easier to demo and inspect during development.

## Testing Strategy

We added a meaningful test suite under [round2/CausalRepair/tests](round2/CausalRepair/tests).

The tests cover different layers of the stack:

- adapter behavior and contract validation,
- environment step behavior,
- reward calculation,
- adapter swapping via `CR_ADAPTER`,
- and a full scripted end-to-end repair episode.

In particular, [round2/CausalRepair/tests/test_full_flow.py](round2/CausalRepair/tests/test_full_flow.py) validates the main product story: starting from a broken world, diagnosing the fault, repairing it, propagating the change, and receiving the expected shaped reward after a successful commit.

That test is valuable because it checks the whole loop instead of only unit-level pieces.

## What We Actually Built

Looking across the repository, the deliverables are:

- a domain-agnostic causal repair environment,
- a stable adapter interface,
- a code-repair benchmark domain with hidden fault injection,
- a second hydraulic adapter to validate generality,
- an OpenEnv/FastAPI server wrapper,
- an LLM inference loop with reward shaping,
- manual verification and REPL tooling,
- a packaging and deployment scaffold,
- and a test suite for both unit and integration behavior.

In other words, we built both the **task** and the **infrastructure around the task**.

## Design Decisions That Matter

Several design choices made the project workable:

### 1. Domain logic lives in adapters

This prevents the environment class from becoming a monolith and makes it possible to reuse the same interaction loop across domains.

### 2. Diagnosis is an explicit action

The agent cannot jump directly from symptom to final answer without first deciding what to inspect. That is closer to real repair workflows.

### 3. Propagation is separate from intervention

Changing the world and evaluating the consequences are treated as different actions. This exposes causal structure and enables richer reward design.

### 4. Rewards are shaped outside the environment core

That keeps the environment logic clean and makes experimentation with incentives easier.

### 5. The project includes both agent and human interfaces

The inference loop, REPL, notebook, and verification script together make the system easier to test, demo, and iterate on.

## Current State and Future Extensions

The current version is intentionally compact. The code-repair domain uses a hardcoded mini-world instead of a real repository and real test runner. That was the right tradeoff for getting a full end-to-end architecture working quickly.

The natural next extensions are clear:

- replace the hardcoded functions with real source files,
- replace in-memory test specs with real discovered tests,
- expand fault injection beyond a single mutation,
- add richer dependency graphs and propagation paths,
- improve the packaged client surface so it fully matches the current action and observation schema,
- and scale evaluation across more repair domains.

## Conclusion

CausalRepair is best understood as a modular benchmark architecture for causal diagnosis and repair. The environment core manages episodes, the adapter defines the domain, the server exposes the environment, and the agent loop turns it into an evaluable task.

What we made here is more than a single environment file. We built a reusable structure for repair-oriented RL experiments, plus the tooling, scripts, packaging, and tests needed to develop against it with confidence.

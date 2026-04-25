# causal_code_repair.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


# ----------------------------------------------------------------------
# Data structures exchanged between LLM and environment
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """What the LLM sees at each step."""
    description: str
    extra: Dict[str, Any] = field(default_factory=dict)



@dataclass(frozen=True)
class Action:
    """
    Action the LLM may take.
    Exactly one of the four types is allowed; unused fields should be None.
    """
    action_type: Literal["diagnose", "intervene", "propagate", "commit_repair"]
    target: Optional[str] = None  # e.g. "line_42", "function_foo", "test_bar"
    value: Optional[Any] = None  # new value / edit for intervene
    rationale: Optional[str] = None  # text justification for commit_repair
    payload: Dict[str, Any] = field(default_factory=dict)



@dataclass(frozen=True)
class StepResult:
    """Result returned by ``step``."""
    observation: Observation
    reward: float  # scalar reward for this transition
    done: bool  # episode finished?
    info: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Abstract environment contract
# ----------------------------------------------------------------------
class BaseCausalCodeRepairEnv(ABC):
    """
    Defines the exact call‑signature that both teammates must respect.
    Implementations provide the deterministic world (code‑base, tests,
    fault injection, reward logic) while keeping the interface stable.
    """

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Observation:
        """
        Start a new episode.

        Parameters
        ----------
        seed : int | None
            Optional seed for reproducible bug injection / program selection.
        options : dict | None
            Extra configuration (e.g., difficulty, language, max edit budget).

        Returns
        -------
        Observation
            Initial observation presented to the LLM.
        """
        ...

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """
        Execute one LLM action and return the resulting observation,
        reward, termination flag, and auxiliary info.

        Parameters
        ----------
        action : Action
            The LLM's chosen action.

        Returns
        -------
        StepResult
            Contains the next observation, scalar reward, done flag,
            and a dict for logging/debugging.
        """
        ...

    # ------------------------------------------------------------------
    # Optional helpers (not strictly required but useful for UI/debug)
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """
        Return a raw, internal representation of the world (e.g., AST,
        current test outcomes, edit distance).  Shape is implementation‑specific.
        """
        raise NotImplementedError

    def render(self, mode: str = "human") -> Optional[str]:
        """
        Optional: produce a human‑readable rendering of the current state.
        Return a string (or None) depending on ``mode``.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Clean up any resources (e.g., delete temporary files, close subprocesses).
        """
        pass


# ----------------------------------------------------------------------
# Example concrete skeleton (env‑person fills in the bodies)
# ----------------------------------------------------------------------
class CausalCodeRepairEnv(BaseCausalCodeRepairEnv):
    """
    Minimal stub that the environment developer can expand.
    Replace the method bodies with the actual code‑repair logic.
    """

    def __init__(
        self,
        *,
        language: str = "python",
        max_edit_budget: int = 3,
        test_timeout_sec: float = 5.0,
        **kwargs: Any,
    ):
        self.language = language
        self.max_edit_budget = max_edit_budget
        self.test_timeout_sec = test_timeout_sec
        # internal state: original source, faulty version, applied edits, etc.
        self._reset_internal_state()

    # ------------------------------------------------------------------
    # Internal helpers (not part of the public contract)
    # ------------------------------------------------------------------
    def _reset_internal_state(self) -> None:
        """Load a fresh program, inject a fault, compute initial failing test."""
        raise NotImplementedError

    def _apply_edit(self, target: str, value: Any) -> None:
        """Mutate the internal copy of the code according to the edit."""
        raise NotImplementedError

    def _run_tests(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Run the test suite on the current code version.
        Returns (all_passed, per_test_details).
        """
        raise NotImplementedError

    def _compute_reward(
        self,
        all_passed: bool,
        per_test_details: Dict[str, Any],
        num_edits: int,
    ) -> float:
        """
        Shape the scalar reward from test outcomes, edit budget, etc.
        Implement the multi‑signal reward you designed.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public contract implementations
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Observation:
        if seed is not None:
            # seed any RNG used for program selection / fault injection
            pass
        self._reset_internal_state()
        obs_description = self._build_initial_description()
        return Observation(description=obs_description, extra={})

    def step(self, action: Action) -> StepResult:
        # 1. Interpret the action
        if action.action_type == "diagnose":
            # LLM asks for info about a target; we return diagnostic data
            diag_info = self._get_diagnostic_info(action.target or "")
            obs = Observation(description=diag_info, extra={})
            reward = -0.1   # small cost for each diagnose call
            done = False
            info = {"diagnose_target": action.target}
        elif action.action_type == "intervene":
            if action.target is None or action.value is None:
                raise ValueError("intervene action requires target and value")
            self._apply_edit(action.target, action.value)
            obs = Observation(description=self._build_state_description(), extra={})
            reward = 0.0    # no immediate reward; reward comes after commit/propagate
            done = False
            info = {"intervene_target": action.target}
        elif action.action_type == "propagate":
            all_passed, details = self._run_tests()
            obs = Observation(
                description=self._format_test_result(all_passed, details),
                extra={"test_details": details},
            )
            # reward is based purely on test outcome (could be shaped)
            reward = 1.0 if all_passed else -0.5
            done = False
            info = {"test_passed": all_passed}
        elif action.action_type == "commit_repair":
            if action.target is None or action.rationale is None:
                raise ValueError("commit_repair requires target and rationale")
            # Final verification
            all_passed, details = self._run_tests()
            num_edits = self._count_edits()
            reward = self._compute_reward(all_passed, details, num_edits)
            done = True
            info = {
                "commit_target": action.target,
                "rationale": action.rationale,
                "test_passed": all_passed,
                "num_edits": num_edits,
            }
            obs = Observation(
                description=self._format_final_outcome(all_passed, details),
                extra={"test_details": details},
            )
        else:
            raise ValueError(f"Unknown action_type: {action.action_type}")

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    # ------------------------------------------------------------------
    # Optional UI / debug helpers (can be stubbed)
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "max_edit_budget": self.max_edit_budget,
            # include whatever internal representation you need
        }

    def render(self, mode: str = "human") -> Optional[str]:
        if mode == "human":
            return self._build_state_description()
        return None

    def close(self) -> None:
        # cleanup temporary files, kill subprocesses, etc.
        pass

    # ------------------------------------------------------------------
    # Placeholder methods – to be implemented by the env‑person
    # ------------------------------------------------------------------
    def _build_initial_description(self) -> str:
        """Return the string shown to the LLM after reset."""
        raise NotImplementedError

    def _build_state_description(self) -> str:
        """Return a description of the current code / test state."""
        raise NotImplementedError

    def _get_diagnostic_info(self, target: str) -> str:
        """Return diagnostic information for ``target`` (e.g., runtime values)."""
        raise NotImplementedError

    def _format_test_result(self, passed: bool, details: Dict[str, Any]) -> str:
        """Human‑readable summary of the last test run."""
        raise NotImplementedError

    def _format_final_outcome(self, passed: bool, details: Dict[str, Any]) -> str:
        """Message shown after a ``commit_repair``."""
        raise NotImplementedError

    def _count_edits(self) -> int:
        """Number of edits applied since the last reset."""
        raise NotImplementedError
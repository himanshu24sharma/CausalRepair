"""
BaseAdapter: the contract every CausalRepair domain adapter must implement.

This file is the frozen interface CausalRepairEnv.py depends on. The core env
(written by Person B) calls exactly seven methods on whichever adapter instance
it holds:

    generate_world, inject_fault, render_observation, diagnose,
    intervene, propagate, check_constraints

Adapters are domain-specific (hydraulic, code, config, ...) but every adapter
exposes the same seven methods so the env stays domain-agnostic.

Mutation semantics (locked across all adapters):
    - generate_world():           pure factory; returns a fresh dict
    - inject_fault, intervene,
      propagate:                  in-place; return None
    - diagnose, render_observation,
      check_constraints:          read-only on world

Observation format (locked across all domains): the .description field of the
returned CausalrepairObservation MUST follow this exact 5-section template, so
the LLM prompt and Person B's regex action parser stay unchanged when adapters
swap.

    DOMAIN: <name>
    STATE:
      <entity>: <value>
      ...
    RULES:
      <entity>: <human-readable description>
      ...
    CONSTRAINTS:
      [OK]/[VIOLATED] <constraint_name>
      ...
    AVAILABLE ACTIONS:
      diagnose("entity_name")
      intervene("entity_name", "value")
      propagate()
      commit_repair("entity_name", "value", rationale="...")
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from models import CausalrepairObservation


class BaseAdapter(ABC):
    domain_name: str = "unknown"

    @abstractmethod
    def generate_world(self) -> Dict[str, Any]:
        """Return a fresh, fault-free world dict for this domain."""
        ...

    @abstractmethod
    def inject_fault(self, world: Dict[str, Any]) -> None:
        """In-place: silently break one rule/value/edge so >=1 constraint becomes False."""
        ...

    @abstractmethod
    def render_observation(self, world: Dict[str, Any]) -> CausalrepairObservation:
        """Pack the 5-section text into a CausalrepairObservation and return it."""
        ...

    @abstractmethod
    def diagnose(self, world: Dict[str, Any], entity: str) -> str:
        """Return human-readable diagnostic info for one entity."""
        ...

    @abstractmethod
    def intervene(self, world: Dict[str, Any], entity: str, value: Any) -> None:
        """In-place: apply the proposed change to `world`."""
        ...

    @abstractmethod
    def propagate(self, world: Dict[str, Any]) -> None:
        """In-place: re-run rules so downstream values reflect the latest state."""
        ...

    @abstractmethod
    def check_constraints(self, world: Dict[str, Any]) -> bool:
        """Return True iff every domain constraint is satisfied on the current world."""
        ...

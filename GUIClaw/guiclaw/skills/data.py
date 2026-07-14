"""
guiclaw.skills.data
===================
Skill and SkillStep dataclasses — the atomic unit of reusable GUI knowledge.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def collect_placeholder_names(value: Any) -> set[str]:
    """Collect ``{{name}}`` placeholders nested inside a skill value."""
    if isinstance(value, str):
        return set(_PLACEHOLDER_RE.findall(value))
    if isinstance(value, dict):
        names: set[str] = set()
        for key, item in value.items():
            names.update(collect_placeholder_names(key))
            names.update(collect_placeholder_names(item))
        return names
    if isinstance(value, (list, tuple, set, frozenset)):
        names: set[str] = set()
        for item in value:
            names.update(collect_placeholder_names(item))
        return names
    return set()


@dataclass(frozen=True)
class SkillStep:
    """One atomic action within a skill sequence.

    Supports two execution modes:
    - Normal mode (``fixed=False``): parameters are resolved by the LLM at
      runtime using ``{{param_name}}`` placeholders.
    - Fixed mode (``fixed=True``): ``fixed_values`` supplies concrete parameter
      values that bypass LLM grounding entirely.
    """

    action_type: str
    target: str
    parameters: dict[str, Any] = field(default_factory=dict)
    valid_state: str | None = None
    state_contract: dict[str, Any] | None = field(
        default=None,
        hash=False,
        compare=False,
    )
    fixed: bool = False
    # hash=False, compare=False: mutable dict inside frozen dataclass would
    # otherwise raise TypeError when the dataclass is hashed (e.g. used in a
    # set or as a dict key).
    fixed_values: dict[str, Any] = field(
        default_factory=dict,
        hash=False,
        compare=False,
    )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "action_type": self.action_type,
            "target": self.target,
        }
        if self.parameters:
            d["parameters"] = self.parameters
        if self.valid_state is not None:
            d["valid_state"] = self.valid_state
        if self.state_contract:
            d["state_contract"] = self.state_contract
        # Only serialise fixed-mode fields when they carry non-default data,
        # keeping the dict compact and backward-compatible.
        if self.fixed:
            d["fixed"] = True
        if self.fixed_values:
            d["fixed_values"] = self.fixed_values
        return d

@dataclass(frozen=True)
class Skill:
    """A reusable, parameterized GUI skill extracted from trajectories.

    Parameters use ``{{param_name}}`` placeholders in step targets/parameters
    that are grounded at execution time.

    """

    skill_id: str
    name: str
    description: str
    app: str
    platform: str
    steps: tuple[SkillStep, ...] = ()
    parameters: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)
    success_count: int = 0
    failure_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "app": self.app,
            "platform": self.platform,
            "steps": [s.to_dict() for s in self.steps],
            "parameters": list(self.parameters),
            "tags": list(self.tags),
            "created_at": self.created_at,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
        }


def compute_confidence(skill: Skill) -> float:
    """Compute skill confidence using Laplace (add-one) smoothing.

    Uses ``(success + 1) / (success + failure + 2)`` to avoid the
    death-spiral where a single failure drives confidence to zero and
    permanently locks out an otherwise correct skill.

    Reference values:
    - No history (0/0) → 0.50  (neutral prior)
    - 1 failure  (0/1) → 0.33  (penalised but recoverable)
    - 1 success  (1/1) → 0.67
    - 2 failures (0/2) → 0.25
    - 5 failures (0/5) → 0.14

    Args:
        skill: The skill whose confidence to compute.

    Returns:
        A float in ``(0.0, 1.0)``.
    """
    return (skill.success_count + 1) / (skill.success_count + skill.failure_count + 2)

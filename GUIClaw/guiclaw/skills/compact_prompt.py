"""Compact skill prompt helpers for MobileWorld-style GUI agents.

This module keeps prompt-only skill selection separate from the heavier
pre-run skill replay path.  Retrieved skills are exposed as ``use_skill``
choices, while curated composite actions are exposed as first-class action
aliases in the general_e2e action table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

USE_SKILL_ACTION_TYPE = "use_skill"
ALWAYS_ON_SKILL_TAG = "compact_action"
ACTION_ALIAS_TAG_PREFIXES = ("action_alias:", "alias:")

COMPACT_SKILL_INSTRUCTIONS = """\
# Reusable Compact GUI Skills
A listed skill may complete the whole task, a useful prefix, or an intermediate subgoal; it
does not need to match the entire task. Use `use_skill` for any matching next part, then
continue the remaining GUI steps normally.

Skill action format:
`{{"action_type":"use_skill","skill_id":"listed_skill_id","arguments":{{"param":"value"}}}}`

Rules:
- If a listed skill matches the task's named app and next part,
  you MUST call `use_skill` before `open_app` or manual navigation; copy its `skill_id` exactly.
- A skill may open or navigate the target app internally.
- Pass only task-provided `arguments`; otherwise use `{{}}`.

Compact skills:
{catalog}
""".strip()

USE_SKILL_ACTION_ROW = (
    '| `use_skill`     | Run a listed skill for a matching task prefix or intermediate subgoal | '
    '`{"action_type":"use_skill","skill_id":"listed_skill_id","arguments":{}}` |'
)

USE_SKILL_DECISION_RULE = (
    "0. Before a manual GUI action, check the compact skill list. Choose `use_skill` "
    "when one can complete the next useful task part, even if it is only a prefix or "
    "intermediate subgoal; continue the remaining steps afterward."
)

# Seed-family models use XML-style function calls instead of the general_e2e JSON
# action table, so the ``use_skill`` capability is described in their own format.
# The catalog body is shared (it is format-agnostic); only the invocation syntax
# differs.  ``__CATALOG__`` is substituted via :func:`format_seed_skill_instructions`
# (plain replace, not str.format, to avoid escaping the JSON braces below).
SEED_COMPACT_SKILL_INSTRUCTIONS = """\
## Optional Compact GUI Skills
Optionally pick ONE listed compact skill as a single action when it clearly matches
the task. If none clearly matches, keep using the normal functions above.

- Additional function available:
{"type": "function", "name": "use_skill", "parameters": {"type": "object", "properties": {"skill_id": {"type": "string", "description": "Exact skill_id copied from the Compact skills list below."}, "arguments": {"type": "string", "description": "JSON object string of skill arguments, e.g. {\\"query\\":\\"hello\\"}. Use {} when no arguments are needed."}}, "required": ["skill_id"]}, "description": "Run a listed compact GUI skill when it clearly matches the requested app/workflow. The skill may open/navigate the target app internally."}

- To call it, use the standard call structure, for example:
<tool_call><function=use_skill><parameter=skill_id>listed_skill_id</parameter><parameter=arguments>{}</parameter></function></tool_call>

Rules:
- Prefer use_skill over manual navigation when a listed skill clearly matches the
  requested app/workflow; copy its skill_id exactly.
- A skill may open/navigate the target app internally, so the target app need not
  already be on screen.
- Fill arguments only with values the task makes obvious; otherwise use {}.

Compact skills:
__CATALOG__"""


def format_seed_skill_instructions(catalog: str) -> str:
    """Render the seed-format compact-skill block for a given catalog body.

    Returns an empty string when there is no catalog so callers can skip the
    injection entirely.
    """
    if not catalog:
        return ""
    return SEED_COMPACT_SKILL_INSTRUCTIONS.replace("__CATALOG__", catalog)

COMPOSITE_ACTION_DEFINITIONS: dict[str, tuple[str, str]] = {
    "click_then_type": (
        "Preferred one-step action for visible text fields: tap a coordinate and type text. Use auto_enter true only for search submission.",
        '`{"action_type":"click_then_type","coordinate":[x,y],"text":"Hello","auto_enter":false}`',
    ),
    "click_multi": (
        "Tap multiple visible coordinates in sequence",
        '`{"action_type":"click_multi","coordinates":[[x1,y1],[x2,y2]]}`',
    ),
}


@dataclass(frozen=True)
class SkillInfo:
    function_name: str
    description: str
    skill_id: str | None = None
    app: str = ""
    parameters: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompositeActionInfo:
    alias: str
    description: str
    example: str


@dataclass(frozen=True)
class CompactPromptParts:
    action_rows: str = ""
    decision_rules: str = ""
    compact_skill_instructions: str = ""
    skill_ids: tuple[str, ...] = ()
    composite_aliases: tuple[str, ...] = ()
    # Raw, format-agnostic catalog body (one line per retrieved skill). Profiles
    # that do not use the general_e2e JSON action table (e.g. ``seed``) wrap this
    # in their own invocation instructions. Empty when no skills were retrieved.
    catalog: str = ""


def skill_info_from_flat_skill(skill: Any) -> SkillInfo:
    return SkillInfo(
        function_name=str(getattr(skill, "name", "") or getattr(skill, "skill_id", "")),
        description=str(getattr(skill, "description", "") or ""),
        skill_id=str(getattr(skill, "skill_id", "") or ""),
        app=str(getattr(skill, "app", "") or ""),
        parameters=tuple(str(param) for param in (getattr(skill, "parameters", ()) or ())),
    )


def build_catalog(skills: list[SkillInfo], *, limit: int | None) -> str:
    selected = skills[:limit] if limit else skills
    return "\n".join(_format_catalog_line(skill) for skill in selected)


def is_shortcut_skill(skill: Any) -> bool:
    tags = {str(tag).lower() for tag in (getattr(skill, "tags", ()) or ())}
    if "shortcut" in tags or "deeplink" in tags or "intent" in tags:
        return True
    skill_id = str(getattr(skill, "skill_id", "") or "")
    if skill_id.startswith("shortcut:"):
        return True
    first_step = skill.steps[0] if getattr(skill, "steps", ()) else None
    return bool(first_step and getattr(first_step, "action_type", "") in {"open_deeplink", "open_intent"})


def is_always_on_skill(skill: Any, tags: tuple[str, ...] | list[str]) -> bool:
    skill_tags = {str(tag).lower() for tag in (getattr(skill, "tags", ()) or ())}
    wanted = {str(tag).lower() for tag in tags if str(tag)}
    return bool(skill_tags & wanted)


def composite_action_infos_from_skills(
    skills: list[Any],
    *,
    tags: tuple[str, ...] | list[str] = (ALWAYS_ON_SKILL_TAG,),
) -> list[CompositeActionInfo]:
    out: list[CompositeActionInfo] = []
    seen: set[str] = set()
    for skill in skills:
        if not is_always_on_skill(skill, tags):
            continue
        alias = composite_alias_from_skill(skill)
        if alias is None or alias in seen:
            continue
        description, example = COMPOSITE_ACTION_DEFINITIONS[alias]
        out.append(
            CompositeActionInfo(
                alias=alias,
                description=description,
                example=example,
            )
        )
        seen.add(alias)
    return out


def composite_alias_from_skill(skill: Any) -> str | None:
    tags = tuple(str(tag) for tag in (getattr(skill, "tags", ()) or ()))
    for tag in tags:
        lowered = tag.lower()
        for prefix in ACTION_ALIAS_TAG_PREFIXES:
            if lowered.startswith(prefix):
                alias = tag[len(prefix):].strip().lower()
                return alias if alias in COMPOSITE_ACTION_DEFINITIONS else None
    name = str(getattr(skill, "name", "") or "").strip().lower()
    return name if name in COMPOSITE_ACTION_DEFINITIONS else None


def build_compact_prompt_parts(
    *,
    retrieved_skills: list[SkillInfo],
    composite_actions: list[CompositeActionInfo],
) -> CompactPromptParts:
    action_rows: list[str] = []
    decision_rules: list[str] = []
    skill_ids: list[str] = []
    composite_aliases: list[str] = []
    catalog = ""

    if retrieved_skills:
        action_rows.append(USE_SKILL_ACTION_ROW)
        decision_rules.append(USE_SKILL_DECISION_RULE)
        skill_ids = [skill.skill_id or skill.function_name for skill in retrieved_skills]
        catalog = build_catalog(retrieved_skills, limit=None)
        compact_skill_instructions = COMPACT_SKILL_INSTRUCTIONS.format(catalog=catalog)
    else:
        compact_skill_instructions = ""

    if composite_actions:
        action_rows.extend(_format_composite_action_row(action) for action in composite_actions)
        composite_aliases = [action.alias for action in composite_actions]
        decision_rules.append(
            "0a. Use listed composite actions only when they exactly match the next "
            "local operation. Prefer `click_multi` when several targets on screen need "
            "the same tap and none opens a confirmation dialog. Prefer `click_then_type` "
            "over separate `click`+`input_text` when tapping a visible text field to enter "
            "known text — unless the field is already focused, existing text must be "
            "cleared, tapping opens a picker to observe first, or the text depends on the "
            "tap result."
        )

    return CompactPromptParts(
        action_rows="\n".join(action_rows),
        decision_rules="\n".join(decision_rules),
        compact_skill_instructions=compact_skill_instructions,
        skill_ids=tuple(skill_ids),
        composite_aliases=tuple(composite_aliases),
        catalog=catalog,
    )


def _format_catalog_line(skill: SkillInfo) -> str:
    parts = [
        f"skill_id={skill.skill_id or skill.function_name}",
        f"skill_name={skill.function_name}",
        f"description={skill.description}",
    ]
    if skill.app:
        parts.append(f"app={skill.app}")
    if skill.parameters:
        parts.append(f"parameters={','.join(skill.parameters)}")
    return "- " + "; ".join(parts)


def _format_composite_action_row(action: CompositeActionInfo) -> str:
    return f"| `{action.alias}` | {action.description} | {action.example} |"


__all__ = [
    "ALWAYS_ON_SKILL_TAG",
    "COMPACT_SKILL_INSTRUCTIONS",
    "COMPOSITE_ACTION_DEFINITIONS",
    "CompactPromptParts",
    "CompositeActionInfo",
    "SEED_COMPACT_SKILL_INSTRUCTIONS",
    "SkillInfo",
    "USE_SKILL_ACTION_ROW",
    "USE_SKILL_ACTION_TYPE",
    "USE_SKILL_DECISION_RULE",
    "build_catalog",
    "format_seed_skill_instructions",
    "build_compact_prompt_parts",
    "composite_action_infos_from_skills",
    "composite_alias_from_skill",
    "is_always_on_skill",
    "is_shortcut_skill",
    "skill_info_from_flat_skill",
]

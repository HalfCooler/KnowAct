"""Tests for scripts/induce_compact_skills.py compact post-processing."""

from __future__ import annotations

from pathlib import Path

import pytest

from guiclaw.skills.data import Skill, SkillStep, collect_placeholder_names
from guiclaw.skills.flat import C, R
from guiclaw.skills.induction import (
    compactify_skill,
)
from guiclaw.skills.induction import (
    has_terminal_action as _has_terminal_action,
)
from scripts.induce_compact_skills import cluster_compact_skills, merge_into_output


def _step(
    action_type: str,
    target: str = "button",
    *,
    valid_state: str | None = "field is visible",
    state_contract: dict | None = None,
    parameters: dict | None = None,
    fixed_values: dict | None = None,
) -> SkillStep:
    return SkillStep(
        action_type=action_type,
        target=target,
        parameters=parameters or {},
        valid_state=valid_state,
        state_contract=state_contract,
        fixed_values=fixed_values or {},
    )


def _skill(
    *steps: SkillStep,
    name: str = "fill_form",
    app: str = "com.gmailclone",
    skill_id: str = "flat:fill_form",
) -> Skill:
    return Skill(
        skill_id=skill_id,
        name=name,
        description="When to use: test",
        app=app,
        platform="android",
        steps=tuple(steps),
        parameters=(),
    )


def _succ(*skills: Skill) -> list[tuple[Skill, bool]]:
    """Wrap skills as success-derived cluster items."""
    return [(s, True) for s in skills]


# ---------------------------------------------------------------------------
# compactify_skill
# ---------------------------------------------------------------------------


class TestCompactifySkill:
    def test_accepts_and_retags(self):
        skill = _skill(
            _step("tap", "To field"),
            _step("input_text", "{{to_email}}"),
            _step("tap", "Subject field"),
            _step("input_text", "{{subject}}"),
        )
        out = compactify_skill(skill, max_steps=7, max_scroll_steps=1)
        assert out is not None
        assert out.tags == ("compact", "compact_extracted")
        assert out.skill_id == "compact:com.gmailclone:fill_form"
        # Guards are preserved verbatim; skipping NL validation is a runtime policy.
        assert [s.valid_state for s in out.steps] == ["field is visible"] * 4

    def test_preserves_step_guards(self):
        contract = C(app="com.gmailclone", required=[R(resource_id="to_field", visible=True)])
        skill = _skill(
            _step("tap", "To field", valid_state="To field is focused", state_contract=contract),
            _step("input_text", "{{to_email}}"),
        )
        out = compactify_skill(skill, max_steps=7, max_scroll_steps=1)
        assert out is not None
        # Both the deterministic contract and the prose valid_state survive intact.
        assert out.steps[0].state_contract == contract
        assert out.steps[0].valid_state == "To field is focused"

    def test_rejects_single_step(self):
        assert compactify_skill(_skill(_step("tap")), max_steps=7, max_scroll_steps=1) is None

    def test_rejects_too_many_steps(self):
        steps = [_step("tap", f"b{i}") for i in range(8)]
        assert compactify_skill(_skill(*steps), max_steps=7, max_scroll_steps=1) is None

    def test_accepts_open_app_as_first_step(self):
        skill = _skill(
            _step("open_app", "com.gmailclone"),
            _step("tap", "compose"),
            _step("input_text", "{{subject}}"),
        )
        assert compactify_skill(skill, max_steps=7, max_scroll_steps=1) is not None

    def test_rejects_open_app_in_middle(self):
        skill = _skill(
            _step("tap", "compose"),
            _step("open_app", "com.other"),
        )
        assert compactify_skill(skill, max_steps=7, max_scroll_steps=1) is None

    def test_scroll_budget(self):
        two_scrolls = _skill(
            _step("scroll", "list"),
            _step("scroll", "list"),
            _step("tap", "item"),
        )
        assert compactify_skill(two_scrolls, max_steps=7, max_scroll_steps=1) is None
        assert compactify_skill(two_scrolls, max_steps=7, max_scroll_steps=2) is not None


# ---------------------------------------------------------------------------
# placeholder extraction
# ---------------------------------------------------------------------------


class TestPlaceholderNames:
    def test_extracts_from_nested_structures(self):
        assert collect_placeholder_names("{{to}}") == {"to"}
        assert collect_placeholder_names({"text": "{{a}}", "k": "{{b}}"}) == {"a", "b"}
        assert collect_placeholder_names(["{{x}}", {"y": "{{y}}"}]) == {"x", "y"}
        assert collect_placeholder_names("literal") == set()


# ---------------------------------------------------------------------------
# cross-trajectory clustering
# ---------------------------------------------------------------------------


class TestClustering:
    def _email_skill(self, skill_id: str) -> Skill:
        # Same structure, different literal targets — must still cluster.
        return _skill(
            _step("tap", "To"),
            _step("input_text", "{{to_email}}"),
            name="fill_email",
            skill_id=skill_id,
        )

    def test_identical_structure_clusters_with_support(self):
        skills = [
            compactify_skill(self._email_skill("flat:a"), max_steps=7, max_scroll_steps=1),
            compactify_skill(self._email_skill("flat:b"), max_steps=7, max_scroll_steps=1),
        ]
        clustered = cluster_compact_skills(_succ(*[s for s in skills if s]), min_support=1)
        assert len(clustered) == 1
        assert clustered[0].success_count == 2

    def test_min_support_filters_singletons(self):
        skill = compactify_skill(self._email_skill("flat:a"), max_steps=7, max_scroll_steps=1)
        assert cluster_compact_skills(_succ(skill), min_support=2) == []
        assert len(cluster_compact_skills(_succ(skill), min_support=1)) == 1

    def test_distinct_structures_do_not_cluster(self):
        a = compactify_skill(self._email_skill("flat:a"), max_steps=7, max_scroll_steps=1)
        b = compactify_skill(
            _skill(
                _step("tap", "search"),
                _step("input_text", "{{q}}"),
                _step("tap", "go"),
                name="search_flow",
                skill_id="flat:c",
            ),
            max_steps=7,
            max_scroll_steps=1,
        )
        clustered = cluster_compact_skills(_succ(a, b), min_support=1)
        assert len(clustered) == 2

    def test_same_sequence_different_literal_target_stays_apart(self):
        # Same app, same action sequence and placeholder name, but different
        # literal controls must NOT collapse (would inflate success_count).
        to_field = compactify_skill(
            _skill(
                _step("tap", "To", valid_state=None),
                _step("input_text", "{{value}}"),
                name="fill_to",
                skill_id="flat:a",
            ),
            max_steps=7,
            max_scroll_steps=1,
        )
        search = compactify_skill(
            _skill(
                _step("tap", "Search", valid_state=None),
                _step("input_text", "{{value}}"),
                name="fill_search",
                skill_id="flat:b",
            ),
            max_steps=7,
            max_scroll_steps=1,
        )
        clustered = cluster_compact_skills(_succ(to_field, search), min_support=1)
        assert len(clustered) == 2
        assert all(s.success_count == 1 for s in clustered)


class TestTerminalActionBackstop:
    def test_detects_terminal_targets(self):
        assert _has_terminal_action(_skill(_step("tap", "Send"), _step("tap", "ok")))
        assert _has_terminal_action(_skill(_step("tap", "Delete message")))
        assert _has_terminal_action(_skill(_step("tap", "Confirm purchase")))

    def test_benign_targets_pass(self):
        assert not _has_terminal_action(_skill(_step("tap", "To"), _step("input_text", "{{q}}")))
        # word-boundary: "sender" / "reorder" must not trip send / order rules
        assert not _has_terminal_action(_skill(_step("tap", "Sender name")))
        assert not _has_terminal_action(_skill(_step("tap", "Reorder list")))

    def test_identifier_style_targets_detected(self):
        # snake_case / camelCase must not evade the backstop
        assert _has_terminal_action(_skill(_step("tap", "send_button")))
        assert _has_terminal_action(_skill(_step("tap", "submitOrder")))


class TestFailureProvenance:
    def _email(self, skill_id: str) -> Skill:
        return compactify_skill(
            _skill(
                _step("tap", "To"),
                _step("input_text", "{{to_email}}"),
                name="fill_email",
                skill_id=skill_id,
            ),
            max_steps=7,
            max_scroll_steps=1,
        )

    def test_failure_only_cluster_dropped(self):
        # A workflow only ever seen failing is never promoted.
        fail = self._email("flat:a")
        assert cluster_compact_skills([(fail, False)], min_support=1) == []

    def test_failure_corroborates_success_without_inflating_count(self):
        # 1 success + 1 failure reaches min_support=2, but success_count stays 1
        # and the cluster is tagged from_failure.
        clustered = cluster_compact_skills(
            [(self._email("flat:a"), True), (self._email("flat:b"), False)],
            min_support=2,
        )
        assert len(clustered) == 1
        assert clustered[0].success_count == 1
        assert "from_failure" in clustered[0].tags

    def test_pure_success_cluster_not_tagged(self):
        clustered = cluster_compact_skills(
            [(self._email("flat:a"), True), (self._email("flat:b"), True)],
            min_support=2,
        )
        assert len(clustered) == 1
        assert "from_failure" not in clustered[0].tags
        assert clustered[0].success_count == 2


# ---------------------------------------------------------------------------
# output merge
# ---------------------------------------------------------------------------


class TestMergeOutput:
    def test_writes_and_merges_monotonic_support(self, tmp_path: Path):
        out = tmp_path / "compact_skills.py"
        skill = compactify_skill(
            _skill(_step("tap", "To"), _step("input_text", "{{to_email}}"), name="fill_email"),
            max_steps=7,
            max_scroll_steps=1,
        )
        clustered = cluster_compact_skills(_succ(skill, skill), min_support=1)  # support=2
        added = merge_into_output(clustered, out)
        assert added == 1
        assert out.exists()

        # Re-merge a lower-support version of the same skill: support must not drop.
        weaker = cluster_compact_skills(_succ(skill), min_support=1)  # support=1
        added2 = merge_into_output(weaker, out)
        assert added2 == 0
        from guiclaw.skills.flat import compile_flat_skills

        compiled = compile_flat_skills(out.read_text(encoding="utf-8"))
        assert not compiled.errors
        assert len(compiled.skills) == 1
        assert compiled.skills[0].success_count == 2

    def test_keeps_shortcuts_before_extracted_skills(self, tmp_path: Path):
        from guiclaw.skills.flat import export_skills_to_source

        shortcut = Skill(
            skill_id="shortcut:dl:com.gmailclone:compose",
            name="open_compose",
            description="Open compose",
            app="com.gmailclone",
            platform="android",
            tags=("shortcut", "deeplink", "validated"),
        )
        out = tmp_path / "skills.py"
        out.write_text(export_skills_to_source([shortcut]), encoding="utf-8")
        extracted = compactify_skill(
            _skill(_step("tap", "To"), _step("input_text", "{{to_email}}")),
            max_steps=7,
            max_scroll_steps=1,
        )

        merge_into_output([extracted], out)

        from guiclaw.skills.flat import compile_flat_skills

        compiled = compile_flat_skills(out.read_text(encoding="utf-8"))
        assert [skill.skill_id for skill in compiled.skills] == [
            shortcut.skill_id,
            extracted.skill_id,
        ]


class TestPackagedCompactInduction:
    def test_success_source_is_compact_and_contributes_support(self) -> None:
        from guiclaw.skills.induction import compactify_skill as compactify_online

        candidate = compactify_online(
            _skill(_step("tap", "Search"), _step("input_text", "{{query}}")),
            is_success=True,
        )

        assert candidate is not None
        assert candidate.tags == ("compact", "compact_extracted")
        assert candidate.skill_id == "compact:com.gmailclone:fill_form"
        assert candidate.success_count == 1

    def test_failure_terminal_action_is_rejected(self) -> None:
        from guiclaw.skills.induction import compactify_skill as compactify_online

        candidate = compactify_online(
            _skill(_step("tap", "Message"), _step("tap", "send_button")),
            is_success=False,
        )

        assert candidate is None

    @pytest.mark.asyncio
    async def test_trace_entry_uses_codegen_result_for_single_app(self, tmp_path: Path) -> None:
        from guiclaw.skills import induction

        codegen_result = type("Codegen", (), {"steps": [object()]})()
        extracted = _skill(_step("tap", "Search"), _step("input_text", "{{query}}"))

        class Extractor:
            def __init__(self) -> None:
                self.seen_result = None

            async def extract_from_codegen_result_multi(
                self,
                result: object,
                *,
                is_success: bool,
            ) -> list[Skill]:
                assert is_success is True
                self.seen_result = result
                return [extracted]

        extractor = Extractor()
        original = induction.codegen_trajectory
        induction.codegen_trajectory = lambda path, subtask_index=1: codegen_result
        try:
            skills = await induction.induce_compact_skills_from_trace(
                extractor,
                tmp_path / "traj.json",
                is_success=True,
                subtask_index=2,
            )
        finally:
            induction.codegen_trajectory = original

        assert extractor.seen_result is codegen_result
        assert len(skills) == 1
        assert skills[0].success_count == 1

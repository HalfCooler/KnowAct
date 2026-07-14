from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    ("guiclaw.grounding", "guiclaw.prompts"),
)
def test_disconnected_compatibility_packages_are_removed(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize(
    ("module_name", "symbol"),
    (
        ("guiclaw.skills.static_selector_filter", "filter_static_controls"),
        ("guiclaw.skills.static_selector_filter", "filter_static_texts"),
        ("guiclaw.skills.static_selector_filter", "filter_static_resource_ids"),
        ("guiclaw.skills.trajectory_codegen", "_first_foreground_app"),
        ("guiclaw.skills.deeplink", "profile_to_skills"),
        ("guiclaw.skills.deeplink", "probe_deep_link"),
        ("guiclaw.skills.deeplink", "probe_deep_intent"),
        ("guiclaw.agents.utils.helpers", "compact_json"),
        ("guiclaw.agents.utils.parsers", "parse_and_check_json_markdown"),
        ("guiclaw.backends.adb", "_is_adb_transport_error"),
        ("guiclaw.skills.executor", "ObservationProvider"),
        ("guiclaw.backends.keycodes", "KeyCodeResolver"),
        ("guiclaw.image_utils", "scale_image_half"),
        ("guiclaw.skills.executor", "_scale_image_half"),
        ("guiclaw.skills.state_contract", "state_contract_overlap"),
        ("guiclaw.skills.state_contract", "merge_state_contracts"),
        ("guiclaw.skills.state_contract", "score_state_contract"),
        ("guiclaw.skills.state_contract", "infer_state_contract"),
        ("guiclaw.skills.state_contract", "_find_selector_for_target"),
        ("guiclaw.skills.state_contract", "_infer_mask_rules"),
        ("guiclaw.skills.state_contract", "_best_target_match"),
        ("guiclaw.skills.state_contract", "_find_selector_for_step"),
        ("guiclaw.skills.state_contract", "_rank_selector_candidates"),
        ("guiclaw.skills.state_contract", "_candidate_selectors"),
        ("guiclaw.skills.state_contract", "_context_grounded_selectors"),
        ("guiclaw.skills.state_contract", "_coordinate_grounded_selectors"),
        ("guiclaw.skills.state_contract", "_structured_selector_labels"),
        ("guiclaw.skills.state_contract", "_selector_from_node"),
        ("guiclaw.skills.state_contract", "_node_labels"),
        ("guiclaw.skills.state_contract", "_step_context_text"),
        ("guiclaw.skills.state_contract", "_context_mentions_label"),
        ("guiclaw.skills.state_contract", "_node_selectors_for_target"),
        ("guiclaw.skills.state_contract", "_selector_candidate"),
        ("guiclaw.skills.state_contract", "_selector_confidence"),
        ("guiclaw.skills.state_contract", "_selector_specificity"),
        ("guiclaw.skills.state_contract", "_iter_observation_extras"),
        ("guiclaw.skills.state_contract", "_inference_extras"),
        ("guiclaw.skills.state_contract", "_extra_from_mapping"),
        ("guiclaw.skills.state_contract", "_infer_mask_rules_from_extras"),
        ("guiclaw.skills.state_contract", "_exact_target_match"),
        ("guiclaw.skills.state_contract", "_selector_identity_key"),
        ("guiclaw.skills.state_contract", "_should_skip_valid_state"),
        ("guiclaw.skills.normalization", "resolve_ios_bundle"),
        ("guiclaw.skills.normalization", "annotate_ios_apps"),
        ("guiclaw.skills.normalization", "resolve_android_package"),
        ("guiclaw.agents.profiles", "normalize_profile_response"),
        ("guiclaw.agent_profiles", "normalize_profile_response"),
    ),
)
def test_approved_dead_symbols_are_removed(module_name: str, symbol: str) -> None:
    module = importlib.import_module(module_name)
    assert not hasattr(module, symbol), f"{module_name}.{symbol} still exists"


def test_flat_skill_library_uses_only_shared_app_filter_normalization() -> None:
    flat = importlib.import_module("guiclaw.skills.flat")
    normalization = importlib.import_module("guiclaw.skills.normalization")

    assert hasattr(normalization, "normalize_app_filter")
    assert not hasattr(flat, "_normalize_app_filter")


def test_skills_package_has_no_eager_compatibility_exports() -> None:
    skills = importlib.import_module("guiclaw.skills")

    assert not hasattr(skills, "SkillLibrary")
    assert not hasattr(skills, "Skill")
    assert not hasattr(skills, "SkillExecutor")


def test_skill_extractor_has_no_step_dict_compatibility_api() -> None:
    extractor = importlib.import_module("guiclaw.skills.extractor")

    assert not hasattr(extractor.SkillExtractor, "extract_from_steps")
    assert not hasattr(extractor.SkillExtractor, "extract_from_steps_multi")
    assert not hasattr(extractor, "_codegen_from_step_dicts")


def test_legacy_skill_schema_fields_and_deserializers_are_removed() -> None:
    data = importlib.import_module("guiclaw.skills.data")

    assert not hasattr(data.Skill, "from_dict")
    assert not hasattr(data.SkillStep, "from_dict")
    assert "preconditions" not in data.Skill.__dataclass_fields__
    assert "success_streak" not in data.Skill.__dataclass_fields__
    assert "failure_streak" not in data.Skill.__dataclass_fields__
    assert "expected_state" not in data.SkillStep.__dataclass_fields__


def test_flat_tag_compatibility_decorator_is_removed() -> None:
    flat = importlib.import_module("guiclaw.skills.flat")

    assert not hasattr(flat, "tag")
    assert "tag" not in flat.CODE_HEADER


@pytest.mark.parametrize(
    "symbol",
    (
        "_find_best_conflict",
        "_heuristic_merge_decision",
        "_merge_skills",
        "_cleanup_superseded_prefixes",
    ),
)
def test_flat_module_does_not_shadow_merger_helpers(symbol: str) -> None:
    flat = importlib.import_module("guiclaw.skills.flat")
    assert not hasattr(flat, symbol)


def test_mobileworld_agents_and_induction_contracts_are_retained() -> None:
    retained_classes = (
        ("guiclaw.agents.implementations.gelab_agent", "GelabAgent"),
        ("guiclaw.agents.implementations.general_e2e_agent", "GeneralE2EAgentMCP"),
        ("guiclaw.agents.implementations.gui_owl_1_5", "GUIOWL15AgentMCP"),
        ("guiclaw.agents.implementations.mai_ui_agent", "MAIUINaivigationAgent"),
        ("guiclaw.agents.implementations.planner_executor", "PlannerExecutorAgentMCP"),
        ("guiclaw.agents.implementations.qwen3vl", "Qwen3VLAgentMCP"),
        ("guiclaw.agents.implementations.seed_agent", "SeedAgent"),
        ("guiclaw.agents.implementations.ui_venus_agent", "VenusNaviAgent"),
    )
    for module_name, symbol in retained_classes:
        assert hasattr(importlib.import_module(module_name), symbol)

    state_contract = importlib.import_module("guiclaw.skills.state_contract")
    trajectory_codegen = importlib.import_module("guiclaw.skills.trajectory_codegen")
    assert hasattr(state_contract, "state_contract_fingerprint")
    assert hasattr(trajectory_codegen, "codegen_trajectory")
    assert hasattr(trajectory_codegen, "codegen_to_extraction_text")

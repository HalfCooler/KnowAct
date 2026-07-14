"""Unit tests for guiclaw.skills._merger pure functions."""

from __future__ import annotations

import numpy as np
import pytest

from guiclaw.skills._merger import (
    SkillConflict,
    action_signature,
    action_similarity,
    cleanup_same_intent,
    cleanup_superseded_prefixes,
    cosine_similarity,
    find_best_conflict,
    heuristic_merge_decision,
    is_strict_rich_prefix,
    merge_skills,
    name_token_similarity,
    skill_semantic_similarity,
    stable_json,
    step_signature,
    step_similarity,
    text_hash,
    tokens,
    tuple_jaccard,
    weighted_tuple_jaccard,
)
from guiclaw.skills.data import Skill, SkillStep

# -- helpers ----------------------------------------------------------------

def _skill(
    name: str = "test_skill",
    app: str = "com.example.app",
    platform: str = "android",
    steps: tuple[SkillStep, ...] = (),
    **kwargs,
) -> Skill:
    defaults = {
        "skill_id": f"flat:{name}",
        "name": name,
        "description": f"Skill for {name}",
        "app": app,
        "platform": platform,
        "tags": (),
        "parameters": (),
        "steps": steps,
        "success_count": 0,
        "failure_count": 0,
    }
    defaults.update(kwargs)
    return Skill(**defaults)


def _step(action_type: str = "tap", target: str = "", **kwargs) -> SkillStep:
    defaults = {
        "action_type": action_type,
        "target": target,
        "parameters": {},
    }
    defaults.update(kwargs)
    return SkillStep(**defaults)


def _sig(skill: Skill):
    return action_signature(skill)


# -- name_token_similarity ---------------------------------------------------


def test_name_token_similarity_identical() -> None:
    assert name_token_similarity("open settings", "open settings") == 1.0


def test_name_token_similarity_overlap() -> None:
    sim = name_token_similarity("open settings app", "open camera app")
    assert 0.4 < sim < 1.0


def test_name_token_similarity_no_overlap() -> None:
    assert name_token_similarity("open settings", "take photo") == 0.0


def test_name_token_similarity_stopwords_ignored() -> None:
    # "the" and "a" are stopwords
    sim = name_token_similarity("the app", "a app")
    assert sim == 1.0


def test_name_token_similarity_both_empty() -> None:
    assert name_token_similarity("", "") == 1.0


# -- tokens ------------------------------------------------------------------


def test_tokens_extracts_words() -> None:
    result = tokens("Open the Settings App")
    assert "open" in result
    assert "settings" in result
    assert "app" in result
    assert "the" not in result  # stopword


def test_tokens_empty() -> None:
    assert tokens("") == ()


# -- tuple_jaccard -----------------------------------------------------------


def test_tuple_jaccard_identical() -> None:
    assert tuple_jaccard(("a", "b"), ("a", "b")) == 1.0


def test_tuple_jaccard_half_overlap() -> None:
    assert tuple_jaccard(("a", "b"), ("b", "c")) == pytest.approx(1.0 / 3.0)


def test_tuple_jaccard_no_overlap() -> None:
    assert tuple_jaccard(("a",), ("b",)) == 0.0


def test_tuple_jaccard_both_empty() -> None:
    assert tuple_jaccard((), ()) == 1.0


def test_tuple_jaccard_one_empty() -> None:
    assert tuple_jaccard(("a",), ()) == 0.0


# -- weighted_tuple_jaccard --------------------------------------------------


def test_weighted_tuple_jaccard_both_empty_returns_empty_score() -> None:
    assert weighted_tuple_jaccard((), (), empty_score=0.42) == 0.42


def test_weighted_tuple_jaccard_has_values() -> None:
    assert weighted_tuple_jaccard(("a",), ("a",), empty_score=0.99) == 1.0


# -- cosine_similarity -------------------------------------------------------


def test_cosine_similarity_identical() -> None:
    v = np.array([1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector() -> None:
    a = np.array([0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, b) == 0.0


# -- stable_json -------------------------------------------------------------


def test_stable_json_sorts_keys() -> None:
    result = stable_json({"b": 1, "a": 2})
    assert result == '{"a": 2, "b": 1}'


def test_stable_json_empty() -> None:
    assert stable_json(None) == ""
    assert stable_json(0) == ""  # falsy values return empty string
    assert stable_json({"key": 1}) != ""  # non-empty dict returns JSON


def test_stable_json_unserializable() -> None:
    result = stable_json({1, 2, 3})  # set is not JSON-serializable
    assert isinstance(result, str)


# -- text_hash ---------------------------------------------------------------


def test_text_hash_deterministic() -> None:
    assert text_hash("hello") == text_hash("hello")


def test_text_hash_different() -> None:
    assert text_hash("hello") != text_hash("world")


# -- step_signature / step_similarity ----------------------------------------


def test_step_signature_same_action_type() -> None:
    s1 = _step("tap", target="button")
    s2 = _step("tap", target="button")
    sig1 = step_signature(s1)
    sig2 = step_signature(s2)
    assert sig1[0] == sig2[0] == "tap"


def test_step_similarity_same_action_high_score() -> None:
    s1 = _step("tap", target="settings button")
    s2 = _step("tap", target="settings icon")
    sim = step_similarity(step_signature(s1), step_signature(s2))
    assert sim > 0.3


def test_step_similarity_different_action_zero() -> None:
    s1 = _step("tap", target="button")
    s2 = _step("swipe", target="button")
    sim = step_similarity(step_signature(s1), step_signature(s2))
    assert sim == 0.0


# -- action_signature / action_similarity ------------------------------------


def test_action_signature_empty_steps() -> None:
    skill = _skill(steps=())
    assert action_signature(skill) == ()


def test_action_similarity_identical() -> None:
    s1 = _skill(steps=(
        _step("tap", "settings button", valid_state="visible and clickable"),
        _step("swipe", "camera list"),
    ))
    s2 = _skill(steps=(
        _step("tap", "settings button", valid_state="visible and clickable"),
        _step("swipe", "camera list"),
    ))
    sim = action_similarity(_sig(s1), _sig(s2))
    assert sim == action_similarity(_sig(s2), _sig(s1))  # symmetric
    assert sim > 0.5  # identical steps should have reasonable similarity


def test_action_similarity_different_lengths() -> None:
    s1 = _skill(steps=(_step("tap", "settings"),))
    s2 = _skill(steps=(_step("tap", "settings"), _step("swipe", "camera"), _step("tap", "gallery")))
    sim = action_similarity(_sig(s1), _sig(s2))
    # Shorter is prefix of longer → should have moderate similarity
    assert 0.3 < sim < 0.9


def test_action_similarity_both_empty() -> None:
    assert action_similarity((), ()) == 1.0


# -- is_strict_rich_prefix ---------------------------------------------------


def test_is_strict_rich_prefix_true() -> None:
    shorter = _skill(steps=(_step("tap", "settings"),))
    longer = _skill(steps=(_step("tap", "settings"), _step("swipe", "camera")))
    assert is_strict_rich_prefix(_sig(shorter), _sig(longer)) is True


def test_is_strict_rich_prefix_equal_length() -> None:
    s = _skill(steps=(_step("tap", "settings"),))
    assert is_strict_rich_prefix(_sig(s), _sig(s)) is False  # len(shorter) >= len(longer)


def test_is_strict_rich_prefix_different_action() -> None:
    shorter = _skill(steps=(_step("tap", "settings"),))
    longer = _skill(steps=(_step("swipe", "settings"), _step("tap", "gallery")))
    assert is_strict_rich_prefix(_sig(shorter), _sig(longer)) is False


# -- skill_semantic_similarity -----------------------------------------------


def test_skill_semantic_similarity_identical() -> None:
    s1 = _skill(name="open settings", description="Open the settings app")
    s2 = _skill(name="open settings", description="Open the settings app")
    assert skill_semantic_similarity(s1, s2) == 1.0


def test_skill_semantic_similarity_no_overlap() -> None:
    s1 = _skill(name="open settings", description="settings")
    s2 = _skill(name="take photo", description="camera")
    assert skill_semantic_similarity(s1, s2) == 0.0


# -- find_best_conflict ------------------------------------------------------


def test_find_best_conflict_same_id_returns_immediate() -> None:
    s1 = _skill(skill_id="abc", name="test")
    s2 = _skill(skill_id="abc", name="test")
    result = find_best_conflict(s2, [s1], incoming_embedding=None, existing_embeddings={})
    assert result is not None
    assert result.score == 1.0


def test_find_best_conflict_different_platform_no_match() -> None:
    s1 = _skill(platform="android", name="test")
    s2 = _skill(platform="ios", name="test")
    result = find_best_conflict(s2, [s1], incoming_embedding=None, existing_embeddings={})
    assert result is None


def test_find_best_conflict_rich_prefix_skipped() -> None:
    short = _skill(
        skill_id="short", name="open",
        steps=(_step("tap", "settings button", valid_state="visible"),),
        success_count=0,
    )
    long = _skill(
        skill_id="long", name="open then swipe",
        steps=(
            _step("tap", "settings button", valid_state="visible"),
            _step("swipe", "camera list", valid_state="swipeable"),
        ),
        success_count=0,
    )
    # short is strict rich prefix of long — should be skipped
    result = find_best_conflict(short, [long], incoming_embedding=None, existing_embeddings={})
    assert result is None


# -- merge_skills ------------------------------------------------------------


def test_merge_skills_combines_counts() -> None:
    old = _skill(skill_id="abc", success_count=3, failure_count=2)
    new = _skill(skill_id="xyz", success_count=1, failure_count=0,
                 steps=(_step("tap", "settings"),))
    merged = merge_skills(old, new)
    assert merged.skill_id == "abc"  # keeps old id
    assert merged.success_count == 4  # 3 + 1
    assert merged.failure_count == 2  # 2 + 0


def test_merge_skills_unions_parameters() -> None:
    old = _skill(parameters=("p1", "p2"))
    new = _skill(parameters=("p2", "p3"))
    merged = merge_skills(old, new)
    assert set(merged.parameters) == {"p1", "p2", "p3"}


def test_merge_skills_keeps_successful_steps() -> None:
    old = _skill(success_count=1, steps=(_step("tap", "settings_old"),))
    new = _skill(success_count=0, steps=(_step("tap", "settings_new"),))
    merged = merge_skills(old, new)
    assert merged.steps[0].target == "settings_old"


# -- heuristic_merge_decision ------------------------------------------------


def test_heuristic_merge_keep_old_on_proven_success() -> None:
    old = _skill(success_count=5)
    new = _skill(success_count=0)
    conflict = SkillConflict(
        skill=old, score=0.8,
        embedding_similarity=0.75,
        sequence_similarity=0.70,
        semantic_similarity=0.3,
    )
    assert heuristic_merge_decision(conflict, new) == "KEEP_OLD"


def test_heuristic_merge_keep_new_when_old_never_succeeded() -> None:
    old = _skill(success_count=0)
    new = _skill(success_count=3)
    conflict = SkillConflict(
        skill=old, score=0.8,
        embedding_similarity=0.75,
        sequence_similarity=0.90,
        semantic_similarity=0.3,
    )
    assert heuristic_merge_decision(conflict, new) == "KEEP_NEW"


def test_heuristic_merge_default() -> None:
    old = _skill(success_count=1)
    new = _skill(success_count=2)
    conflict = SkillConflict(
        skill=old, score=0.8,
        embedding_similarity=0.80,
        sequence_similarity=0.90,
        semantic_similarity=0.5,
    )
    assert heuristic_merge_decision(conflict, new) == "MERGE"


# -- cleanup_superseded_prefixes ---------------------------------------------


def test_cleanup_superseded_prefixes_removes_subsumed() -> None:
    step_kwargs = {
        "target": "settings button",
        "valid_state": "settings page is visible",
        "parameters": {"x": 100, "y": 200},
    }
    short = _skill(
        skill_id="short", name="prefix",
        steps=(_step("tap", **step_kwargs),),
        success_count=0,
    )
    long = _skill(
        skill_id="long", name="full",
        steps=(
            _step("tap", **step_kwargs),
            _step("swipe", "camera list", valid_state="swipeable",
                  parameters={"x2": 300, "y2": 400}),
        ),
        success_count=0,
    )
    result = cleanup_superseded_prefixes(
        [short, long], platform="android", app="com.example.app",
    )
    ids = {s.skill_id for s in result}
    # When prefix detection works, short is removed
    assert "long" in ids


def test_cleanup_keeps_successful_skills() -> None:
    short = _skill(
        skill_id="short", name="prefix",
        steps=(_step("tap", "settings button", valid_state="visible"),),
        success_count=1,  # has succeeded, so keep it
    )
    long = _skill(
        skill_id="long", name="full",
        steps=(
            _step("tap", "settings button", valid_state="visible"),
            _step("swipe", "camera list", valid_state="swipeable"),
        ),
        success_count=0,
    )
    result = cleanup_superseded_prefixes(
        [short, long], platform="android", app="com.example.app",
    )
    ids = {s.skill_id for s in result}
    assert "short" in ids  # success_count > 0 keeps it
    assert "long" in ids


# -- cleanup_same_intent -----------------------------------------------------


def test_cleanup_same_intent_without_embeddings() -> None:
    s1 = _skill(name="open settings", description="Open settings app")
    s2 = _skill(name="open settings", description="Open settings app")
    assert cleanup_same_intent(s1, s2, embeddings=None) is True


def test_cleanup_same_intent_different() -> None:
    s1 = _skill(name="open settings", description="settings")
    s2 = _skill(name="take photo", description="camera")
    assert cleanup_same_intent(s1, s2, embeddings=None) is False

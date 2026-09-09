from __future__ import annotations

import pytest

from guiclaw.action import Action
from guiclaw.planner_escalation import (
    canonicalize_repeat_judge_model,
    parse_repeat_verdict,
    should_judge_repeat,
    ui_tree_difference_ratio,
)

_HOME_TREE = [
    {"resource_id": "com.app:id/title", "text": "Home", "class": "TextView", "enabled": True},
    {"resource_id": "com.app:id/search", "text": "Search", "class": "Button", "clickable": True},
    {"resource_id": "com.app:id/item", "text": "Result", "class": "TextView", "clickable": True},
]
_SEARCH_TREE = [
    {"resource_id": "com.app:id/query", "text": "Query", "class": "EditText", "enabled": True},
    {"resource_id": "com.app:id/submit", "text": "Go", "class": "Button", "clickable": True},
    {"resource_id": "com.app:id/hint", "text": "Type here", "class": "TextView"},
]


def test_should_judge_repeat_requires_same_type_and_similar_ui_tree() -> None:
    tap = Action(action_type="tap", x=10, y=20)
    other_tap = Action(action_type="tap", x=80, y=90)
    typed = Action(action_type="input_text", text="hello")
    done = Action(action_type="done", status="success")
    similar_tree = list(_HOME_TREE)
    jittered_tree = [
        {**node, "bounds": "[0,0][10,10]", "focused": True}
        for node in _HOME_TREE
    ]

    assert should_judge_repeat(None, tap, previous_tree=similar_tree, current_tree=similar_tree) is False
    assert should_judge_repeat(
        tap, other_tap, previous_tree=similar_tree, current_tree=similar_tree
    ) is True
    assert should_judge_repeat(
        tap, other_tap, previous_tree=similar_tree, current_tree=jittered_tree
    ) is True
    assert should_judge_repeat(
        tap, other_tap, previous_tree=similar_tree, current_tree=_SEARCH_TREE
    ) is False
    assert should_judge_repeat(tap, other_tap) is False
    assert should_judge_repeat(
        tap, typed, previous_tree=similar_tree, current_tree=similar_tree
    ) is False
    assert should_judge_repeat(
        done, done, previous_tree=similar_tree, current_tree=similar_tree
    ) is False


def test_ui_tree_difference_ratio_ignores_bounds_and_missing_trees() -> None:
    identical = list(_HOME_TREE)
    jittered = [{**node, "bounds": "[1,1][2,2]"} for node in _HOME_TREE]
    one_replaced = [
        _HOME_TREE[0],
        _HOME_TREE[1],
        {"resource_id": "com.app:id/other", "text": "Other", "class": "TextView"},
    ]

    assert ui_tree_difference_ratio(identical, identical) == 0.0
    assert ui_tree_difference_ratio(identical, jittered) == 0.0
    assert ui_tree_difference_ratio(identical, one_replaced) == pytest.approx(1 / 3)
    assert ui_tree_difference_ratio(None, identical) is None
    assert ui_tree_difference_ratio([], identical) is None
    assert ui_tree_difference_ratio(identical, _SEARCH_TREE) == 1.0


def test_canonicalize_repeat_judge_model() -> None:
    assert canonicalize_repeat_judge_model(None) == "small"
    assert canonicalize_repeat_judge_model("LARGE") == "large"
    with pytest.raises(ValueError, match="repeat_judge_model"):
        canonicalize_repeat_judge_model("medium")


@pytest.mark.parametrize(
    ("content", "repeated"),
    [
        ('{"repeat": true, "reason": "same button"}', True),
        ('{"repeat": false, "reason": "new field"}', False),
        ("```json\n{\"repeat\": true, \"reason\": \"loop\"}\n```", True),
        ('sure\n{"repeat": false, "reason": "ok"}', False),
        ('{"repeat": "yes"}', True),
        ("true", True),
        ("false", False),
        ("", False),
        ("I am not sure", False),
    ],
)
def test_parse_repeat_verdict(content: str, repeated: bool) -> None:
    assert parse_repeat_verdict(content).repeated is repeated

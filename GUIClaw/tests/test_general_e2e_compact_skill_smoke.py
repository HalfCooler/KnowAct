from __future__ import annotations

import pytest

from guiclaw.skills.compact_prompt import (
    build_catalog,
    is_shortcut_skill,
    skill_info_from_flat_skill,
)
from guiclaw.skills.data import Skill, SkillStep
from guiclaw.skills.flat import FlatSkillLibrary


def test_shortcut_skill_catalog_includes_id_and_parameters() -> None:
    skill = Skill(
        skill_id="shortcut:dl:tv.danmaku.bili:search",
        name="bili_search",
        description="B站搜索视频",
        app="tv.danmaku.bili",
        platform="android",
        tags=("shortcut", "deeplink", "validated"),
        parameters=("query",),
        steps=(
            SkillStep(
                action_type="open_deeplink",
                target="bilibili://search?keyword={{query}}",
                parameters={
                    "text": "bilibili://search?keyword={{query}}",
                    "package": "tv.danmaku.bili",
                },
                valid_state="B站搜索视频",
            ),
        ),
    )

    catalog = build_catalog([skill_info_from_flat_skill(skill)], limit=None)

    assert "skill_id=shortcut:dl:tv.danmaku.bili:search" in catalog
    assert "parameters=query" in catalog


@pytest.mark.asyncio
async def test_retrieve_skill_infos_can_filter_to_shortcut_skills(tmp_path) -> None:
    library = FlatSkillLibrary(store_dir=tmp_path)
    library.add(
        Skill(
            skill_id="manual:tap:tv.danmaku.bili",
            name="manual_bili_step",
            description="B站普通点击步骤",
            app="tv.danmaku.bili",
            platform="android",
            steps=(SkillStep(action_type="tap", target="搜索按钮"),),
        )
    )
    library.add(
        Skill(
            skill_id="shortcut:dl:tv.danmaku.bili:search",
            name="bili_search",
            description="B站搜索视频",
            app="tv.danmaku.bili",
            platform="android",
            tags=("shortcut", "deeplink", "validated"),
            parameters=("query",),
            steps=(
                SkillStep(
                    action_type="open_deeplink",
                    target="bilibili://search?keyword={{query}}",
                    parameters={
                        "text": "bilibili://search?keyword={{query}}",
                        "package": "tv.danmaku.bili",
                    },
                    valid_state="B站搜索视频",
                ),
            ),
        )
    )

    results = await library.search(
        "在B站搜索敢杀我的马",
        platform="android",
        app="tv.danmaku.bili",
        top_k=3,
    )
    shortcut_ids = [skill.skill_id for skill, _score in results if is_shortcut_skill(skill)]

    assert shortcut_ids == ["shortcut:dl:tv.danmaku.bili:search"]

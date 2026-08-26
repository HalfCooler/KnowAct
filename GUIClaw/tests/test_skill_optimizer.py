from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import guiclaw.skills._merger as _merger
from guiclaw.skills.data import Skill, SkillStep
from guiclaw.skills.flat import (
    SKILL_EMBEDDINGS_CACHE_VERSION,
    SKILL_EMBEDDINGS_FILENAME,
    SKILL_EMBEDDINGS_META_FILENAME,
    FlatSkillLibrary,
    _skill_search_text,
)
from guiclaw.skills.optimizer import (
    SkillOptimizationStore,
    apply_skill_optimizations,
    optimize_shortcut_prefixes,
)


def _deeplink(skill_id: str = "shortcut:music:search", description: str = "Search music") -> Skill:
    return Skill(
        skill_id=skill_id,
        name="music_search",
        description=description,
        app="com.music",
        platform="android",
        parameters=("query",),
        tags=("shortcut", "deeplink"),
        steps=(SkillStep("open_deeplink", "music://search?q={{query}}"),),
    )


def _ui_skill(skill_id: str = "compact:music:search_and_play") -> Skill:
    return Skill(
        skill_id=skill_id,
        name="search_and_play",
        description="Search music and play the first result",
        app="com.music",
        platform="android",
        parameters=("query",),
        tags=("compact",),
        steps=(
            SkillStep("open_app", "com.music"),
            SkillStep("tap", "search"),
            SkillStep("input_text", "query", {"text": "{{query}}"}),
            SkillStep("tap", "first result"),
        ),
    )


class _LLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return SimpleNamespace(content=json.dumps(self.payload), usage={"prompt_tokens": 10})


@pytest.mark.asyncio
async def test_optimizer_replaces_ui_prefix(tmp_path: Path) -> None:
    llm = _LLM(
        {
            "action": "replace_prefix",
            "candidate_id": "candidate_0",
            "replace_prefix_steps": 3,
            "argument_map": {"query": "query"},
            "reason": "deeplink reaches search results",
        }
    )
    skills = [_deeplink(), _ui_skill()]

    report = await optimize_shortcut_prefixes(
        skills,
        embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        store_dir=tmp_path,
        llm=llm,
    )

    assert report["activated"] == [_ui_skill().skill_id]
    effective = apply_skill_optimizations(skills, SkillOptimizationStore(tmp_path))
    optimized = next(skill for skill in effective if skill.skill_id == _ui_skill().skill_id)
    assert [step.action_type for step in optimized.steps] == ["open_deeplink", "tap"]
    recipe = SkillOptimizationStore(tmp_path).read()["recipes"][_ui_skill().skill_id]
    assert "base_skill_hash" not in recipe
    assert "shortcut_skill_hash" not in recipe


@pytest.mark.asyncio
async def test_optimizer_deletes_fully_covered_ui_skill(tmp_path: Path) -> None:
    deleted: list[str] = []
    llm = _LLM(
        {
            "action": "delete",
            "candidate_id": "candidate_0",
            "argument_map": {"query": "query"},
            "reason": "deeplink fully covers this skill",
        }
    )

    report = await optimize_shortcut_prefixes(
        [_deeplink(), _ui_skill()],
        embeddings=np.ones((2, 2), dtype=np.float32),
        store_dir=tmp_path,
        llm=llm,
        delete_skill=lambda skill_id: deleted.append(skill_id) is None,
    )

    assert deleted == [_ui_skill().skill_id]
    assert report["deleted"] == [_ui_skill().skill_id]
    assert not SkillOptimizationStore(tmp_path).path.exists()


@pytest.mark.asyncio
async def test_optimizer_keep_does_not_modify_store(tmp_path: Path) -> None:
    report = await optimize_shortcut_prefixes(
        [_deeplink(), _ui_skill()],
        embeddings=np.ones((2, 2), dtype=np.float32),
        store_dir=tmp_path,
        llm=_LLM({"action": "keep"}),
    )

    assert report["accepted"] == []
    assert not SkillOptimizationStore(tmp_path).path.exists()


@pytest.mark.asyncio
async def test_optimizer_filters_only_by_app_and_retrieves_top_k(tmp_path: Path) -> None:
    other = Skill(
        skill_id="shortcut:other:search",
        name="other_search",
        description="Search another app",
        app="com.other",
        platform="ios",
        tags=("shortcut", "deeplink"),
        steps=(SkillStep("open_deeplink", "other://search"),),
    )
    second = _deeplink("shortcut:music:history", "Open history")
    llm = _LLM({"action": "keep"})

    report = await optimize_shortcut_prefixes(
        [_deeplink(), second, other, _ui_skill()],
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        store_dir=tmp_path,
        llm=llm,
        embedding_top_k=1,
    )

    assert report["candidate_count"] == 1
    assert report["retrieval"][0]["shortcut_skill_id"] == _deeplink().skill_id
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "replace_prefix" in prompt and "delete" in prompt and "keep" in prompt
    assert "Assume every deeplink executes precisely" in prompt


@pytest.mark.asyncio
async def test_flat_library_reuses_cached_embeddings_without_provider_call(
    tmp_path: Path,
) -> None:
    library = FlatSkillLibrary(
        store_dir=tmp_path,
        embedding_provider=SimpleNamespace(
            embed=lambda _texts: (_ for _ in ()).throw(AssertionError("provider called"))
        ),
        embedding_signature="test-embedding",
        merge_llm=_LLM({"action": "keep"}),
    )
    library.add(_deeplink())
    library.add(_ui_skill())
    skills = library._repository.list_all()
    np.save(tmp_path / SKILL_EMBEDDINGS_FILENAME, np.ones((2, 2), dtype=np.float32))
    (tmp_path / SKILL_EMBEDDINGS_META_FILENAME).write_text(
        json.dumps(
            {
                "version": SKILL_EMBEDDINGS_CACHE_VERSION,
                "embedding_signature": "test-embedding",
                "records": [
                    {
                        "skill_id": skill.skill_id,
                        "search_text_hash": _merger.text_hash(_skill_search_text(skill)),
                        "embedding_row": index,
                    }
                    for index, skill in enumerate(skills)
                ],
            }
        ),
        encoding="utf-8",
    )

    report = await library.optimize_shortcut_prefixes(persist=False)

    assert report["candidate_count"] == 1


@pytest.mark.asyncio
async def test_flat_library_embeds_only_missing_rows(tmp_path: Path) -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def embed(self, texts: list[str]) -> np.ndarray:
            self.calls.append(texts)
            return np.ones((len(texts), 2), dtype=np.float32)

    provider = Provider()
    library = FlatSkillLibrary(
        store_dir=tmp_path,
        embedding_provider=provider,
        embedding_signature="test-embedding",
        merge_llm=_LLM({"action": "keep"}),
    )
    library.add(_deeplink())
    library.add(_ui_skill())
    skills = library._repository.list_all()
    np.save(tmp_path / SKILL_EMBEDDINGS_FILENAME, np.ones((1, 2), dtype=np.float32))
    (tmp_path / SKILL_EMBEDDINGS_META_FILENAME).write_text(
        json.dumps(
            {
                "version": SKILL_EMBEDDINGS_CACHE_VERSION,
                "embedding_signature": "test-embedding",
                "records": [
                    {
                        "skill_id": skills[0].skill_id,
                        "search_text_hash": _merger.text_hash(_skill_search_text(skills[0])),
                        "embedding_row": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    await library.optimize_shortcut_prefixes(persist=False)

    assert len(provider.calls) == 1
    assert len(provider.calls[0]) == 1

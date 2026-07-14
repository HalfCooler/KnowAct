"""
guiclaw.skills.flat
===================
Minimal Python-backed GUI skills.

The only persistent skill source is ``skills.py``.  It contains declarative
``@skill`` functions made of awaited ``action(...)`` calls.  No graph cache,
JSON skill bucket, transition evidence, or legacy store is involved.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

import guiclaw.skills._merger as _merger
from guiclaw.action import normalize_action_type
from guiclaw.skills.data import Skill, SkillStep, collect_placeholder_names
from guiclaw.skills.normalization import (
    DEFAULT_GUI_SKILL_STORE_ROOT,
    annotate_android_apps,
    is_unknown_app_identifier,
    normalize_app_filter,
    normalize_app_identifier,
    normalize_skill_app,
)
from guiclaw.skills.state_contract import normalize_state_contract

logger = logging.getLogger(__name__)
_STORE_LOCKS: dict[Path, threading.RLock] = {}
_STORE_LOCKS_GUARD = threading.Lock()

CANONICAL_SKILLS_FILENAME = "skills.py"
DEFAULT_SKILLS_STORE_DIR = DEFAULT_GUI_SKILL_STORE_ROOT
DEFAULT_SKILLS_SOURCE_PATH = DEFAULT_SKILLS_STORE_DIR / CANONICAL_SKILLS_FILENAME
SKILL_EMBEDDINGS_FILENAME = "skills_embeddings.npy"
SKILL_EMBEDDINGS_META_FILENAME = "skills_embeddings_meta.json"
SKILL_EMBEDDINGS_CACHE_VERSION = 1
SKILL_FEEDBACK_FILENAME = "skill_feedback.json"
SKILL_FEEDBACK_VERSION = 1
CODE_HEADER = "from guiclaw.skills.flat import C, R, action, skill"

_STATE_FLAGS = ("visible", "clickable", "enabled", "focused", "scrollable")
_SELECTOR_KEYS = ("text", "content_desc", "resource_id", "class", "xpath")
_R_ALLOWED_KEYS = frozenset((*_STATE_FLAGS, *_SELECTOR_KEYS, "class_"))
_C_ALLOWED_KEYS = frozenset(("required", "forbidden", "app", "activity"))
_PLACEHOLDER_RE = re.compile(r"\{\{([^{}]+)\}\}")


def _store_lock(store_dir: Path) -> threading.RLock:
    key = store_dir.expanduser().resolve(strict=False)
    with _STORE_LOCKS_GUARD:
        lock = _STORE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _STORE_LOCKS[key] = lock
        return lock


@dataclass(frozen=True)
class FlatAction:
    action_type: str
    target: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    valid_state: str | None = None
    state_contract: dict[str, Any] | None = None
    fixed: bool = False
    fixed_values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FlatSkillMeta:
    name: str
    app: str
    platform: str
    tags: tuple[str, ...] = ()
    skill_id: str | None = None
    description: str = ""
    created_at: float | None = None
    success_count: int = 0
    failure_count: int = 0


@dataclass(frozen=True)
class FlatCompileResult:
    skills: tuple[Skill, ...] = ()
    errors: tuple[str, ...] = ()


class UnsupportedSkillSourceError(ValueError):
    """Raised when declarative skill source uses unsupported Python."""


def R(**kwargs: Any) -> dict[str, Any]:  # noqa: N802
    unsupported = tuple(str(key) for key in kwargs if str(key) not in _R_ALLOWED_KEYS)
    if unsupported:
        raise ValueError(
            "unsupported R() field: "
            + ", ".join(unsupported)
            + "; R() only accepts selector/state fields."
        )
    selector: dict[str, Any] = {}
    state_flags: list[str] = []
    for key, value in kwargs.items():
        normalized_key = "class" if key == "class_" else key
        if normalized_key in _STATE_FLAGS:
            if value:
                state_flags.append(normalized_key)
            continue
        if normalized_key in _SELECTOR_KEYS and value is not None:
            selector[normalized_key] = value
    element: dict[str, Any] = {"selector": selector}
    if state_flags:
        element["state"] = state_flags
    return element


def C(  # noqa: N802
    *,
    required: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    forbidden: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    app: str | None = None,
    activity: str | None = None,
) -> dict[str, Any] | None:
    anchor: dict[str, Any] = {}
    if app:
        anchor["app_package"] = app
    if activity:
        anchor["activity_class"] = activity
    return normalize_state_contract(
        {
            "anchor": anchor,
            "signature": {
                "required": list(required or ()),
                "forbidden": list(forbidden or ()),
            },
        }
    )


def _contract_from_dict(contract: dict[str, Any]) -> dict[str, Any] | None:
    return normalize_state_contract(contract)


C.from_dict = _contract_from_dict  # type: ignore[attr-defined]


def _contract_with_anchor(
    contract: dict[str, Any] | None,
    *,
    app: str | None = None,
    activity: str | None = None,
) -> dict[str, Any] | None:
    raw: dict[str, Any] = dict(contract or {})
    anchor = dict(raw.get("anchor") or {})
    if app and not anchor.get("app_package"):
        anchor["app_package"] = app
    if activity and not anchor.get("activity_class"):
        anchor["activity_class"] = activity
    if anchor:
        raw["anchor"] = anchor
    return normalize_state_contract(raw)


def skill(
    *,
    app: str,
    platform: str,
    tags: list[str] | tuple[str, ...] | None = None,
    skill_id: str | None = None,
    name: str | None = None,
    description: str = "",
    created_at: float | None = None,
    success_count: int = 0,
    failure_count: int = 0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        merged_tags = tuple(dict.fromkeys((*getattr(func, "__guiclaw_tags__", ()), *(tags or ()))))
        setattr(
            func,
            "__guiclaw_skill__",
            FlatSkillMeta(
                name=name or func.__name__,
                app=app,
                platform=platform,
                tags=merged_tags,
                skill_id=skill_id,
                description=description,
                created_at=created_at,
                success_count=success_count,
                failure_count=failure_count,
            ),
        )
        setattr(func, "__guiclaw_tags__", merged_tags)
        return func

    return decorator


async def action(action_type: str, target: str = "", **parameters: Any) -> FlatAction:
    valid_state = parameters.pop("valid_state", None)
    state_contract = parameters.pop("state_contract", None)
    fixed = bool(parameters.pop("fixed", False))
    fixed_values = parameters.pop("fixed_values", {}) or {}
    explicit_parameters = parameters.pop("parameters", None)
    if isinstance(explicit_parameters, dict):
        parameters.update(explicit_parameters)
    return FlatAction(
        action_type=action_type,
        target=target,
        parameters=parameters,
        valid_state=valid_state,
        state_contract=normalize_state_contract(state_contract),
        fixed=fixed,
        fixed_values=dict(fixed_values),
    )


def compile_flat_skills(source: str) -> FlatCompileResult:
    try:
        tree = ast.parse(source or "")
    except SyntaxError as exc:
        return FlatCompileResult(errors=(f"syntax error: {exc}",))

    errors = _validate_source_ast(tree)
    if errors:
        return FlatCompileResult(errors=tuple(errors))

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }
    skills: list[Skill] = []
    for func in functions.values():
        if not isinstance(func, ast.AsyncFunctionDef) or not _has_decorator(func, "skill"):
            continue
        try:
            meta = _decorator_kwargs(func, "skill")
            app = str(meta.get("app") or "")
            platform = str(meta.get("platform") or "unknown")
            steps = _extract_steps(func, functions, stack=(), bindings=_self_bindings(func))
            steps = _anchor_skill_step_contracts(steps, app=app)
            skill_kwargs: dict[str, Any] = {
                "skill_id": str(meta.get("skill_id") or f"flat:{func.name}"),
                "name": str(meta.get("name") or func.name),
                "description": str(meta.get("description") or ""),
                "app": normalize_app_identifier(platform, app),
                "platform": platform,
                "tags": tuple(str(t) for t in (meta.get("tags") or ())),
                "parameters": _used_step_parameters(func, steps),
                "steps": steps,
            }
            if meta.get("created_at") is not None:
                skill_kwargs["created_at"] = float(meta["created_at"])
            for count_field in (
                "success_count",
                "failure_count",
            ):
                if count_field in meta:
                    skill_kwargs[count_field] = int(meta[count_field])
            skill_obj = Skill(
                **skill_kwargs,
            )
            skills.append(normalize_skill_app(skill_obj))
        except UnsupportedSkillSourceError as exc:
            errors.append(str(exc))
    if errors:
        return FlatCompileResult(errors=tuple(errors))
    return FlatCompileResult(skills=tuple(skills))


class FlatSkillRepository:
    """Manage the canonical ``skills.py`` source file."""

    def __init__(self, store_dir: Path) -> None:
        self.store_dir = Path(store_dir).expanduser()
        self.source_path = self.store_dir / CANONICAL_SKILLS_FILENAME

    def read_source(self) -> str:
        if not self.source_path.exists():
            return CODE_HEADER + "\n"
        return self.source_path.read_text(encoding="utf-8")

    def list_all(self, *, platform: str | None = None, app: str | None = None) -> list[Skill]:
        result = compile_flat_skills(self.read_source())
        if result.errors:
            logger.warning("Cannot list flat skills: %s", result.errors)
            return []
        normalized_app = normalize_app_filter(platform, app)
        return [
            skill
            for skill in result.skills
            if (platform is None or skill.platform == platform)
            and (normalized_app is None or skill.app == normalized_app)
        ]

    def add(self, skill_obj: Skill) -> str:
        skills = self.list_all()
        replaced = False
        updated: list[Skill] = []
        for existing in skills:
            if existing.skill_id == skill_obj.skill_id:
                updated.append(normalize_skill_app(skill_obj))
                replaced = True
            else:
                updated.append(existing)
        if not replaced:
            updated.append(normalize_skill_app(skill_obj))
        self._write_atomic(export_skills_to_source(updated))
        return skill_obj.skill_id

    def replace_all(self, skills: list[Skill] | tuple[Skill, ...]) -> None:
        self._write_atomic(
            export_skills_to_source([normalize_skill_app(skill) for skill in skills])
        )

    def update(self, skill_id: str, updated_skill: Skill) -> bool:
        skills = self.list_all()
        found = False
        updated: list[Skill] = []
        for existing in skills:
            if existing.skill_id == skill_id:
                updated.append(replace(normalize_skill_app(updated_skill), skill_id=skill_id))
                found = True
            else:
                updated.append(existing)
        if found:
            self._write_atomic(export_skills_to_source(updated))
        return found

    def remove(self, skill_id: str) -> bool:
        skills = self.list_all()
        kept = [skill for skill in skills if skill.skill_id != skill_id]
        if len(kept) == len(skills):
            return False
        self._write_atomic(export_skills_to_source(kept))
        return True

    def _write_atomic(self, source: str) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.source_path.name}.",
            suffix=".tmp",
            dir=str(self.store_dir),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(source.rstrip() + "\n")
            os.replace(tmp_name, self.source_path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


class FlatSkillLibrary:
    """Search adapter over ``skills.py`` flat skills."""

    def __init__(
        self,
        *,
        store_dir: Path,
        embedding_provider: Any | None = None,
        merge_llm: Any | None = None,
        embedding_signature: str | None = None,
    ) -> None:
        self.store_dir = Path(store_dir).expanduser()
        self.embedding_provider = embedding_provider
        self.merge_llm = merge_llm
        self.embedding_signature = embedding_signature
        self._repository = FlatSkillRepository(self.store_dir)
        self._source_mtime: float | None = _mtime(self._repository.source_path)
        self._store_lock = _store_lock(self.store_dir)

    def refresh_if_stale(self) -> bool:
        current = _mtime(self._repository.source_path)
        changed = current != self._source_mtime
        self._source_mtime = current
        return changed

    def load_all(self) -> None:
        self.refresh_if_stale()

    def list_all(self, *, platform: str | None = None, app: str | None = None) -> list[Skill]:
        return self._repository.list_all(platform=platform, app=app)

    def count(self) -> int:
        return len(self.list_all())

    def add(self, skill_obj: Skill) -> str:
        with self._store_lock:
            self._source_mtime = None
            return self._repository.add(self._normalize_skill(skill_obj))

    async def add_or_merge(self, skill_obj: Skill) -> tuple[str, str | None]:
        skill_obj = self._normalize_skill(skill_obj)
        if _is_unknown_app(skill_obj.app):
            logger.warning("Reject skill %s: app is unknown", skill_obj.name)
            return "REJECT_UNKNOWN_APP", None
        with self._store_lock:
            skills_for_embedding = self._repository.list_all()
        incoming_embedding = await self._embed_skill_for_conflict(skill_obj, skills_for_embedding)
        with self._store_lock:
            skills = self._repository.list_all()
            embeddings = self._cached_skill_embedding_map(skills)
            if incoming_embedding is not None:
                embeddings[skill_obj.skill_id] = incoming_embedding
            existing_same_id = next(
                (skill for skill in skills if skill.skill_id == skill_obj.skill_id), None
            )
            if existing_same_id is not None:
                updated = self._replace_in_list(skills, existing_same_id.skill_id, skill_obj)
                updated = _merger.cleanup_superseded_prefixes(
                    updated, skill_obj.platform, skill_obj.app, embeddings=embeddings
                )
                self._write_skills(updated)
                self._prune_feedback_for_skills(updated)
                return "KEEP_NEW", existing_same_id.skill_id

            conflict = _merger.find_best_conflict(
                skill_obj,
                skills,
                incoming_embedding=incoming_embedding,
                existing_embeddings=embeddings,
            )
            if conflict is None:
                updated = [*skills, skill_obj]
                updated = _merger.cleanup_superseded_prefixes(
                    updated, skill_obj.platform, skill_obj.app, embeddings=embeddings
                )
                self._write_skills(updated)
                self._prune_feedback_for_skills(updated)
                return "ADD", skill_obj.skill_id

            decision = _merger.heuristic_merge_decision(conflict, skill_obj)
            if decision == "MERGE":
                merged = _merger.merge_skills(conflict.skill, skill_obj)
                if incoming_embedding is not None:
                    embeddings[merged.skill_id] = incoming_embedding
                updated = self._replace_in_list(skills, conflict.skill.skill_id, merged)
                updated = _merger.cleanup_superseded_prefixes(
                    updated, merged.platform, merged.app, embeddings=embeddings
                )
                self._write_skills(updated)
                self._merge_feedback_records(
                    source_skill_id=skill_obj.skill_id, target_skill_id=merged.skill_id
                )
                self._prune_feedback_for_skills(updated)
                return "MERGE", merged.skill_id
            if decision == "KEEP_NEW":
                updated = [
                    self._normalize_skill(skill_obj)
                    if skill.skill_id == conflict.skill.skill_id
                    else skill
                    for skill in skills
                ]
                updated = _merger.cleanup_superseded_prefixes(
                    updated, skill_obj.platform, skill_obj.app, embeddings=embeddings
                )
                self._write_skills(updated)
                self._merge_feedback_records(
                    source_skill_id=conflict.skill.skill_id, target_skill_id=skill_obj.skill_id
                )
                self._prune_feedback_for_skills(updated)
                return "KEEP_NEW", skill_obj.skill_id
            return "KEEP_OLD", conflict.skill.skill_id

    async def _embed_skill_for_conflict(
        self, skill_obj: Skill, existing: list[Skill]
    ) -> np.ndarray | None:
        if self.embedding_provider is None:
            return None
        if existing:
            await self._ensure_skill_embeddings(existing)
        vector = await self.embedding_provider.embed([_skill_search_text(skill_obj)])
        if vector is None or len(vector) == 0:
            return None
        return np.asarray(vector[0], dtype=np.float32)

    def _cached_skill_embedding_map(self, skills: list[Skill]) -> dict[str, np.ndarray]:
        cached_meta, cached_embeddings = self._load_skill_embedding_cache()
        if (
            not cached_meta
            or cached_embeddings is None
            or cached_meta["embedding_signature"] != self.embedding_signature
        ):
            return {}
        current_keys = {
            (skill.skill_id, _merger.text_hash(_skill_search_text(skill))) for skill in skills
        }
        out: dict[str, np.ndarray] = {}
        for record in cached_meta["records"]:
            key = (record["skill_id"], record["search_text_hash"])
            if key in current_keys:
                out[str(record["skill_id"])] = cached_embeddings[int(record["embedding_row"])]
        return out

    @staticmethod
    def _replace_in_list(skills: list[Skill], skill_id: str, updated_skill: Skill) -> list[Skill]:
        return [
            replace(FlatSkillLibrary._normalize_skill(updated_skill), skill_id=skill_id)
            if skill.skill_id == skill_id
            else skill
            for skill in skills
        ]

    def _write_skills(self, skills: list[Skill]) -> None:
        self._source_mtime = None
        self._repository.replace_all(skills)

    @staticmethod
    def _normalize_skill(skill_obj: Skill) -> Skill:
        skill_obj = normalize_skill_app(skill_obj)
        normalized_steps: list[SkillStep] = []
        changed = False
        for step in skill_obj.steps:
            normalized_contract = normalize_state_contract(step.state_contract)
            if normalized_contract != step.state_contract:
                step = replace(step, state_contract=normalized_contract)
                changed = True
            normalized_steps.append(step)
        if changed:
            skill_obj = replace(skill_obj, steps=tuple(normalized_steps))
        return skill_obj

    async def search(
        self,
        query: str,
        *,
        platform: str | None = None,
        app: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[Skill, float]]:
        if not query.strip() or top_k <= 0:
            return []
        skills = self.list_all()
        normalized_app = normalize_app_filter(platform, app)
        candidate_pairs = [
            (index, skill)
            for index, skill in enumerate(skills)
            if (platform is None or skill.platform == platform)
            and (normalized_app is None or skill.app == normalized_app)
        ]
        candidate_positions = [index for index, _skill in candidate_pairs]
        candidates = [skill for _index, skill in candidate_pairs]
        n = len(candidates)
        if n == 0:
            return []

        from guiclaw.memory.retrieval import _BM25Index

        bm25 = _BM25Index()
        bm25.build([_skill_search_text(skill) for skill in candidates])
        bm25_scores = np.array(bm25.score(query), dtype=np.float32)

        if self.embedding_provider is not None:
            emb_scores = await self._embedding_scores(query, skills, candidate_positions)
            # BM25 rank bonus — boosts candidates that match keywords, no score normalisation
            bm25_order = np.argsort(np.argsort(-bm25_scores)).astype(np.float32)
            bm25_bonus = np.where(bm25_order < 3, (3.0 - bm25_order) * 0.03, 0.0)
            blended = emb_scores + bm25_bonus
        else:
            blended = bm25_scores

        ranked = np.argsort(-blended)
        results: list[tuple[Skill, float]] = []
        for idx in ranked:
            score = float(blended[int(idx)])
            if score <= 0:
                break
            results.append((candidates[int(idx)], score))
            if len(results) >= top_k:
                break
        return results

    async def _embedding_scores(
        self,
        query: str,
        skills: list[Skill],
        candidate_positions: list[int],
    ) -> np.ndarray:
        import faiss

        embeddings = await self._ensure_skill_embeddings(skills)
        candidate_embs = np.ascontiguousarray(embeddings[candidate_positions], dtype=np.float32)
        faiss.normalize_L2(candidate_embs)
        idx = faiss.IndexFlatIP(candidate_embs.shape[1])
        idx.add(candidate_embs)

        q_emb = await self.embedding_provider.embed([query])
        q_vec = np.ascontiguousarray(np.asarray(q_emb[0], dtype=np.float32).reshape(1, -1))
        faiss.normalize_L2(q_vec)
        raw_scores, raw_indices = idx.search(q_vec, len(candidate_positions))
        # Reorder FAISS output (sorted by score desc) back to candidate order
        n = len(candidate_positions)
        scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            idx_i = int(raw_indices[0, i])
            if 0 <= idx_i < n:
                scores[idx_i] = float(raw_scores[0, i])
        return scores

    async def _ensure_skill_embeddings(self, skills: list[Skill]) -> np.ndarray:
        current_records = [
            {
                "skill_id": skill.skill_id,
                "search_text_hash": _merger.text_hash(_skill_search_text(skill)),
            }
            for skill in skills
        ]
        cached_meta, cached_embeddings = self._load_skill_embedding_cache()
        reusable: dict[tuple[str, str], np.ndarray] = {}
        if (
            cached_meta
            and cached_embeddings is not None
            and cached_meta["embedding_signature"] == self.embedding_signature
        ):
            for record in cached_meta["records"]:
                key = (record["skill_id"], record["search_text_hash"])
                reusable[key] = cached_embeddings[int(record["embedding_row"])]

        rows: list[np.ndarray | None] = []
        missing_texts: list[str] = []
        missing_positions: list[int] = []
        for position, (skill, record) in enumerate(zip(skills, current_records, strict=True)):
            key = (record["skill_id"], record["search_text_hash"])
            if key in reusable:
                rows.append(reusable[key])
                continue
            rows.append(None)
            missing_positions.append(position)
            missing_texts.append(_skill_search_text(skill))

        if missing_texts:
            embedded = np.asarray(
                await self.embedding_provider.embed(missing_texts), dtype=np.float32
            )
            for row_index, position in enumerate(missing_positions):
                rows[position] = embedded[row_index]

        embeddings = np.vstack([np.asarray(row, dtype=np.float32) for row in rows])
        should_write = (
            missing_texts
            or not cached_meta
            or cached_embeddings is None
            or cached_meta["embedding_signature"] != self.embedding_signature
            or _cache_record_keys(cached_meta["records"]) != _cache_record_keys(current_records)
        )
        if should_write:
            meta = {
                "version": SKILL_EMBEDDINGS_CACHE_VERSION,
                "embedding_signature": self.embedding_signature,
                "records": [
                    {**record, "embedding_row": row} for row, record in enumerate(current_records)
                ],
            }
            self._write_skill_embedding_cache(embeddings, meta)
        return embeddings

    def _load_skill_embedding_cache(self) -> tuple[dict[str, Any] | None, np.ndarray | None]:
        meta_path = self.store_dir / SKILL_EMBEDDINGS_META_FILENAME
        embeddings_path = self.store_dir / SKILL_EMBEDDINGS_FILENAME
        if not meta_path.exists() or not embeddings_path.exists():
            return None, None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        embeddings = np.load(embeddings_path)
        if meta["version"] != SKILL_EMBEDDINGS_CACHE_VERSION:
            return None, None
        return meta, np.asarray(embeddings, dtype=np.float32)

    def _write_skill_embedding_cache(self, embeddings: np.ndarray, meta: dict[str, Any]) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        embeddings_path = self.store_dir / SKILL_EMBEDDINGS_FILENAME
        meta_path = self.store_dir / SKILL_EMBEDDINGS_META_FILENAME

        fd, tmp_embeddings = tempfile.mkstemp(
            prefix=f".{embeddings_path.name}.",
            suffix=".npy",
            dir=str(self.store_dir),
        )
        os.close(fd)
        try:
            np.save(tmp_embeddings, np.asarray(embeddings, dtype=np.float32))
            os.replace(tmp_embeddings, embeddings_path)
        finally:
            try:
                os.unlink(tmp_embeddings)
            except FileNotFoundError:
                pass

        fd, tmp_meta = tempfile.mkstemp(
            prefix=f".{meta_path.name}.",
            suffix=".tmp",
            dir=str(self.store_dir),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_meta, meta_path)
        finally:
            try:
                os.unlink(tmp_meta)
            except FileNotFoundError:
                pass

    def get(self, skill_id: str) -> Skill | None:
        for skill_obj in self.list_all():
            if skill_obj.skill_id == skill_id:
                return skill_obj
        return None

    def feedback_for_skill(self, skill_id: str) -> dict[str, Any]:
        feedback = self._load_feedback()
        record = feedback.get("skills", {}).get(skill_id)
        return dict(record) if isinstance(record, dict) else {}

    def record_feedback(
        self,
        skill_id: str,
        *,
        task: str | None = None,
        failure_case: dict[str, Any] | None = None,
        status: str | None = None,
        evolved: bool = False,
        timestamp: float | None = None,
    ) -> None:
        if not skill_id:
            return
        with self._store_lock:
            feedback = self._load_feedback()
            skills = feedback.setdefault("skills", {})
            record = skills.setdefault(skill_id, {})
            record["skill_id"] = skill_id
            record["last_updated_at"] = float(timestamp if timestamp is not None else _time_now())
            if task:
                tasks = list(record.get("negative_tasks") or [])
                if task not in tasks:
                    tasks.append(task)
                record["negative_tasks"] = tasks[-20:]
            if failure_case:
                reason = _feedback_failure_reason(failure_case)
                counts = dict(record.get("failure_counts") or {})
                counts[reason] = int(counts.get(reason, 0)) + 1
                record["failure_counts"] = counts
                record["last_failure_case"] = failure_case
                record["last_failure_at"] = record["last_updated_at"]
            if status:
                record["last_evolution_status"] = status
            if evolved:
                record["evolution_count"] = int(record.get("evolution_count") or 0) + 1
            self._write_feedback(feedback)

    def update(self, skill_id: str, updated_skill: Skill) -> bool:
        with self._store_lock:
            self._source_mtime = None
            return self._repository.update(skill_id, self._normalize_skill(updated_skill))

    def remove(self, skill_id: str) -> bool:
        with self._store_lock:
            self._source_mtime = None
            removed = self._repository.remove(skill_id)
            if removed:
                feedback = self._load_feedback()
                skills = feedback.get("skills")
                if isinstance(skills, dict) and skill_id in skills:
                    del skills[skill_id]
                    self._write_feedback(feedback)
            return removed

    def _load_feedback(self) -> dict[str, Any]:
        path = self.store_dir / SKILL_FEEDBACK_FILENAME
        if not path.exists():
            return {"version": SKILL_FEEDBACK_VERSION, "skills": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": SKILL_FEEDBACK_VERSION, "skills": {}}
        if not isinstance(data, dict):
            return {"version": SKILL_FEEDBACK_VERSION, "skills": {}}
        if not isinstance(data.get("skills"), dict):
            data["skills"] = {}
        data["version"] = SKILL_FEEDBACK_VERSION
        return data

    def _write_feedback(self, feedback: dict[str, Any]) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        path = self.store_dir / SKILL_FEEDBACK_FILENAME
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(self.store_dir),
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(feedback, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_name, path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    def _merge_feedback_records(self, *, source_skill_id: str, target_skill_id: str) -> None:
        if not source_skill_id or not target_skill_id or source_skill_id == target_skill_id:
            return
        feedback = self._load_feedback()
        skills = feedback.get("skills")
        if not isinstance(skills, dict) or source_skill_id not in skills:
            return
        source = dict(skills.pop(source_skill_id) or {})
        target = dict(skills.get(target_skill_id) or {})
        target["skill_id"] = target_skill_id
        target["negative_tasks"] = _merge_unique_tail(
            list(target.get("negative_tasks") or []),
            list(source.get("negative_tasks") or []),
            limit=20,
        )
        counts = dict(target.get("failure_counts") or {})
        for key, value in dict(source.get("failure_counts") or {}).items():
            counts[str(key)] = int(counts.get(str(key), 0)) + int(value or 0)
        if counts:
            target["failure_counts"] = counts
        for key in (
            "last_failure_case",
            "last_failure_at",
            "last_evolution_status",
            "last_updated_at",
        ):
            if source.get(key) is not None:
                target[key] = source[key]
        target["evolution_count"] = int(target.get("evolution_count") or 0) + int(
            source.get("evolution_count") or 0
        )
        skills[target_skill_id] = target
        self._write_feedback(feedback)

    def _prune_feedback_for_skills(self, skills: list[Skill]) -> None:
        feedback = self._load_feedback()
        records = feedback.get("skills")
        if not isinstance(records, dict):
            return
        active_ids = {skill.skill_id for skill in skills}
        removed = [skill_id for skill_id in records if skill_id not in active_ids]
        if not removed:
            return
        for skill_id in removed:
            del records[skill_id]
        self._write_feedback(feedback)


def export_skills_to_source(skills: list[Skill] | tuple[Skill, ...]) -> str:
    skills = order_skills_for_source(skills)
    lines: list[str] = [CODE_HEADER, "", ""]
    names = _stable_function_names(skills)
    for skill_obj in skills:
        func_name = names[skill_obj.skill_id]
        decorator_parts = [
            f"app={_code_literal(skill_obj.app)}",
            f"platform={_code_literal(skill_obj.platform)}",
            f"tags={_code_literal(list(skill_obj.tags))}",
            f"skill_id={_code_literal(skill_obj.skill_id)}",
            f"name={_code_literal(skill_obj.name)}",
        ]
        if skill_obj.description:
            decorator_parts.append(f"description={_code_literal(skill_obj.description)}")
        decorator_parts.append(f"created_at={_code_literal(skill_obj.created_at)}")
        if skill_obj.success_count:
            decorator_parts.append(f"success_count={skill_obj.success_count}")
        if skill_obj.failure_count:
            decorator_parts.append(f"failure_count={skill_obj.failure_count}")
        lines.append(f"@skill({', '.join(decorator_parts)})")
        placeholder_map = _parameter_placeholder_map(skill_obj.parameters)
        parameters = [placeholder_map[str(parameter)] for parameter in skill_obj.parameters]
        signature = ", ".join(["device", *parameters])
        lines.append(f"async def {func_name}({signature}):")
        if skill_obj.steps:
            for step in skill_obj.steps:
                lines.append(f"    {_action_call_source(step, placeholder_map)}")
        else:
            lines.append("    pass")
        lines.extend(["", ""])
    return "\n".join(lines).rstrip() + "\n"


def order_skills_for_source(skills: list[Skill] | tuple[Skill, ...]) -> list[Skill]:
    """Return deterministic source order with shortcuts before extracted skills."""

    def sort_key(skill_obj: Skill) -> tuple[Any, ...]:
        tags = {str(tag).casefold() for tag in skill_obj.tags}
        is_shortcut = "shortcut" in tags or skill_obj.skill_id.casefold().startswith("shortcut:")
        return (
            0 if is_shortcut else 1,
            -skill_obj.success_count,
            skill_obj.app.casefold(),
            skill_obj.name.casefold(),
            skill_obj.skill_id.casefold(),
        )

    return sorted(skills, key=sort_key)


def _validate_source_ast(tree: ast.Module) -> list[str]:
    errors: list[str] = []
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }
    function_names = set(functions)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            if node.module != "guiclaw.skills.flat":
                errors.append(f"unsupported import: {node.module}")
            continue
        if isinstance(node, ast.Import):
            errors.append("only from guiclaw.skills.flat imports are allowed")
            continue
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        errors.append(f"unsupported top-level statement: {type(node).__name__}")

    blocked_names = {"eval", "exec", "open", "subprocess", "os", "sys", "adb"}
    blocked_attrs = {"backend", "env", "adb"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            root_name = call_name.split(".", 1)[0]
            if root_name in blocked_names:
                errors.append(f"unsafe call is not allowed: {call_name}")
        if isinstance(node, ast.Attribute):
            root = _attribute_root(node)
            if root in blocked_names or node.attr in blocked_attrs:
                errors.append(f"direct backend/env/adb access is not allowed: {ast.unparse(node)}")

    for func in functions.values():
        if _has_decorator(func, "skill"):
            if not isinstance(func, ast.AsyncFunctionDef):
                errors.append(f"skill function must be async: {func.name}")
            first_arg = func.args.args[0].arg if func.args.args else None
            if first_arg != "device":
                errors.append(f"skill function first argument must be device: {func.name}")
        if isinstance(func, ast.AsyncFunctionDef):
            _validate_function_body(func, function_names, errors)
    return errors


def _validate_function_body(
    func: ast.AsyncFunctionDef,
    function_names: set[str],
    errors: list[str],
) -> None:
    allowed_nested_calls = {"C", "C.from_dict", "R"}
    for stmt in func.body:
        if isinstance(stmt, ast.Pass):
            continue
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Await):
            for call in (node for node in ast.walk(stmt) if isinstance(node, ast.Call)):
                call_name = _call_name(call.func)
                if call_name == "action" or call_name in function_names:
                    errors.append(f"{func.name} must await {call_name}(...)")
            continue
        call = stmt.value.value
        if not isinstance(call, ast.Call):
            errors.append(f"{func.name} awaits a non-call expression")
            continue
        call_name = _call_name(call.func)
        if call_name == "action":
            for nested in ast.walk(call):
                if nested is call or not isinstance(nested, ast.Call):
                    continue
                nested_name = _call_name(nested.func)
                if nested_name not in allowed_nested_calls:
                    errors.append(f"{func.name} contains unsupported nested call: {nested_name}")
            continue
        if call_name in function_names:
            continue
        errors.append(f"{func.name} calls unknown function: {call_name}")


def _extract_steps(
    func: ast.AsyncFunctionDef,
    functions: dict[str, ast.AST],
    *,
    stack: tuple[str, ...],
    bindings: dict[str, ast.AST],
) -> tuple[SkillStep, ...]:
    if func.name in stack:
        cycle = " -> ".join((*stack, func.name))
        raise UnsupportedSkillSourceError(f"recursive helper call: {cycle}")
    steps: list[SkillStep] = []
    for stmt in func.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Await):
            continue
        call = stmt.value.value
        if not isinstance(call, ast.Call):
            continue
        call_name = _call_name(call.func)
        if call_name == "action":
            steps.append(_skill_step_from_action_call(call, bindings))
            continue
        callee = functions.get(call_name)
        if isinstance(callee, ast.AsyncFunctionDef):
            steps.extend(
                _extract_steps(
                    callee,
                    functions,
                    stack=(*stack, func.name),
                    bindings=_bind_call_arguments(callee, call, bindings),
                )
            )
    return tuple(steps)


def _skill_step_from_action_call(call: ast.Call, bindings: dict[str, ast.AST]) -> SkillStep:
    action_type = normalize_action_type(str(_literal_value(call.args[0]) if call.args else ""))
    target = ""
    parameters: dict[str, Any] = {}
    valid_state: str | None = None
    state_contract: dict[str, Any] | None = None
    fixed = False
    fixed_values: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        if kw.arg == "target":
            target = str(_literal_or_placeholder(kw.value, bindings))
            continue
        if kw.arg == "valid_state":
            valid_state = str(_literal_or_placeholder(kw.value, bindings))
            continue
        if kw.arg == "state_contract":
            state_contract = _contract_from_ast(kw.value, bindings)
            continue
        if kw.arg == "fixed":
            fixed = bool(_literal_or_placeholder(kw.value, bindings))
            continue
        if kw.arg == "fixed_values":
            fixed_value = _literal_or_placeholder(kw.value, bindings)
            fixed_values = dict(fixed_value or {})
            continue
        if kw.arg == "parameters":
            explicit_parameters = _literal_or_placeholder(kw.value, bindings)
            if isinstance(explicit_parameters, dict):
                parameters.update(explicit_parameters)
            continue
        parameters[kw.arg] = _literal_or_placeholder(kw.value, bindings)
    return SkillStep(
        action_type=action_type,
        target=target,
        parameters=parameters,
        valid_state=valid_state,
        state_contract=state_contract,
        fixed=fixed,
        fixed_values=fixed_values,
    )


def _contract_from_ast(node: ast.AST, bindings: dict[str, ast.AST]) -> dict[str, Any] | None:
    if not isinstance(node, ast.Call):
        return normalize_state_contract(_literal_value(node))
    call_name = _call_name(node.func)
    if call_name == "C.from_dict":
        if not node.args:
            return None
        return normalize_state_contract(_literal_value(node.args[0]))
    if call_name != "C":
        return None
    if any(kw.arg is None for kw in node.keywords):
        raise UnsupportedSkillSourceError("C() does not support **kwargs")
    unsupported = tuple(
        str(kw.arg) for kw in node.keywords if kw.arg is not None and kw.arg not in _C_ALLOWED_KEYS
    )
    if unsupported:
        raise UnsupportedSkillSourceError(f"unsupported C() field: {', '.join(unsupported)}")
    kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
    required = _selector_list_from_ast(kwargs.get("required"), bindings)
    forbidden = _selector_list_from_ast(kwargs.get("forbidden"), bindings)
    app = _literal_or_placeholder(kwargs["app"], bindings) if "app" in kwargs else None
    activity = (
        _literal_or_placeholder(kwargs["activity"], bindings) if "activity" in kwargs else None
    )
    return C(required=required, forbidden=forbidden, app=app, activity=activity)


def _selector_list_from_ast(
    node: ast.AST | None, bindings: dict[str, ast.AST]
) -> list[dict[str, Any]]:
    if node is None:
        return []
    if not isinstance(node, (ast.List, ast.Tuple)):
        value = _literal_value(node)
        return list(value or ())
    selectors: list[dict[str, Any]] = []
    for element in node.elts:
        if isinstance(element, ast.Call) and _call_name(element.func) == "R":
            selectors.append(_selector_from_r_call(element, bindings))
        else:
            selectors.append(_literal_value(element))
    return selectors


def _selector_from_r_call(call: ast.Call, bindings: dict[str, ast.AST]) -> dict[str, Any]:
    if any(kw.arg is None for kw in call.keywords):
        raise UnsupportedSkillSourceError("R() does not support **kwargs")
    unsupported = tuple(
        str(kw.arg)
        for kw in call.keywords
        if kw.arg is not None and str(kw.arg) not in _R_ALLOWED_KEYS
    )
    if unsupported:
        raise UnsupportedSkillSourceError(f"unsupported R() field: {', '.join(unsupported)}")
    kwargs = {
        kw.arg: _literal_or_placeholder(kw.value, bindings)
        for kw in call.keywords
        if kw.arg is not None
    }
    try:
        return R(**kwargs)
    except ValueError as exc:
        raise UnsupportedSkillSourceError(str(exc)) from exc


def _anchor_skill_step_contracts(
    steps: tuple[SkillStep, ...], *, app: str | None
) -> tuple[SkillStep, ...]:
    if not app:
        return steps
    anchored_steps: list[SkillStep] = []
    for step in steps:
        if not step.state_contract:
            anchored_steps.append(step)
            continue
        anchored_contract = _contract_with_anchor(step.state_contract, app=app)
        if anchored_contract != step.state_contract:
            step = replace(step, state_contract=anchored_contract)
        anchored_steps.append(step)
    return tuple(anchored_steps)


def _literal_or_placeholder(
    node: ast.AST,
    bindings: dict[str, ast.AST],
    *,
    seen: frozenset[str] = frozenset(),
) -> Any:
    if isinstance(node, ast.Name):
        if node.id in bindings and node.id not in seen:
            return _literal_or_placeholder(bindings[node.id], bindings, seen=seen | {node.id})
        return f"{{{{{node.id}}}}}"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_or_placeholder(node.left, bindings, seen=seen)
        right = _literal_or_placeholder(node.right, bindings, seen=seen)
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}{right}"
        raise UnsupportedSkillSourceError(f"unsupported expression: {ast.unparse(node)}")
    return _literal_value(node)


def _literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as exc:
        raise UnsupportedSkillSourceError(f"unsupported expression: {ast.unparse(node)}") from exc


def _self_bindings(func: ast.AsyncFunctionDef) -> dict[str, ast.AST]:
    return {arg.arg: ast.Name(id=arg.arg, ctx=ast.Load()) for arg in func.args.args}


def _bind_call_arguments(
    callee: ast.AsyncFunctionDef,
    call: ast.Call,
    caller_bindings: dict[str, ast.AST],
) -> dict[str, ast.AST]:
    bindings = _self_bindings(callee)
    for arg_def, arg_value in zip(callee.args.args, call.args, strict=False):
        bindings[arg_def.arg] = _resolve_bound_ast(arg_value, caller_bindings)
    for kw in call.keywords:
        if kw.arg is None:
            continue
        bindings[kw.arg] = _resolve_bound_ast(kw.value, caller_bindings)
    return bindings


def _resolve_bound_ast(node: ast.AST, bindings: dict[str, ast.AST]) -> ast.AST:
    if isinstance(node, ast.Name) and node.id in bindings:
        return bindings[node.id]
    return node


def _decorator_kwargs(func: ast.AsyncFunctionDef | ast.FunctionDef, name: str) -> dict[str, Any]:
    for decorator in func.decorator_list:
        if isinstance(decorator, ast.Call) and _call_name(decorator.func) == name:
            return {
                kw.arg: _literal_value(kw.value) for kw in decorator.keywords if kw.arg is not None
            }
    return {}


def _has_decorator(func: ast.AsyncFunctionDef | ast.FunctionDef, name: str) -> bool:
    return any(
        (isinstance(decorator, ast.Call) and _call_name(decorator.func) == name)
        or (isinstance(decorator, ast.Name) and decorator.id == name)
        for decorator in func.decorator_list
    )


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ast.unparse(node)


def _attribute_root(node: ast.Attribute) -> str:
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return ""


def _used_step_parameters(
    func: ast.AsyncFunctionDef, steps: tuple[SkillStep, ...]
) -> tuple[str, ...]:
    declared = tuple(arg.arg for arg in func.args.args[1:])
    if not declared:
        return ()
    used = collect_placeholder_names(
        [
            {
                "target": step.target,
                "parameters": step.parameters,
                "valid_state": step.valid_state,
                "state_contract": step.state_contract,
                "fixed_values": step.fixed_values,
            }
            for step in steps
        ]
    )
    return tuple(name for name in declared if name in used)


def _skill_search_text(skill_obj: Skill) -> str:
    app_alias_text = (
        " ".join(annotate_android_apps([skill_obj.app])) if skill_obj.platform == "android" else ""
    )
    step_text = " ".join(
        " ".join(
            [
                step.action_type,
                step.target,
                " ".join(str(k) for k in step.parameters.keys()),
                " ".join(str(v) for v in step.parameters.values()),
                step.valid_state or "",
                _merger.stable_json(step.state_contract),
            ]
        )
        for step in skill_obj.steps
    )
    base_text = " ".join(
        [
            skill_obj.name,
            skill_obj.description,
            skill_obj.app,
            app_alias_text,
            skill_obj.platform,
            " ".join(skill_obj.tags),
            step_text,
        ]
    )
    return " ".join([base_text, _retrieval_alias_text(base_text)])


def _retrieval_alias_text(text: str) -> str:
    lowered = text.lower()
    aliases: list[str] = []
    if any(token in lowered for token in ("search", "query", "input_text")):
        aliases.extend(["搜索", "查询"])
    if any(token in lowered for token in ("play", "playback", "watch", "video")):
        aliases.extend(["播放", "视频"])
    if any(token in lowered for token in ("open_app", "open ")):
        aliases.extend(["打开", "启动"])
    return " ".join(dict.fromkeys(aliases))


def _cache_record_keys(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(str(record["skill_id"]), str(record["search_text_hash"])) for record in records]


def _is_unknown_app(app: str) -> bool:
    return is_unknown_app_identifier(app)


def _feedback_failure_reason(failure_case: dict[str, Any]) -> str:
    error = failure_case.get("execution_error") or failure_case.get("failure_error")
    if isinstance(error, str) and error.strip():
        return error.strip()[:120]
    target = failure_case.get("failed_target")
    if isinstance(target, str) and target.strip():
        return f"failed_target:{target.strip()[:80]}"
    return "unknown"


def _merge_unique_tail(left: list[Any], right: list[Any], *, limit: int) -> list[Any]:
    merged: list[Any] = []
    for item in [*left, *right]:
        if item and item not in merged:
            merged.append(item)
    return merged[-limit:]


def _time_now() -> float:
    return time.time()


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _stable_function_names(skills: list[Skill] | tuple[Skill, ...]) -> dict[str, str]:
    used: set[str] = set()
    names: dict[str, str] = {}
    for skill_obj in skills:
        base = _safe_identifier(skill_obj.name or skill_obj.skill_id or "skill")
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        names[skill_obj.skill_id] = candidate
    return names


def _safe_identifier(value: str) -> str:
    text = re.sub(r"\W+", "_", value.strip().lower()).strip("_")
    if not text:
        text = "skill"
    if text[0].isdigit():
        text = f"skill_{text}"
    if text in {"class", "def", "return", "async", "await", "from", "import"}:
        text = f"{text}_skill"
    return text


def _parameter_placeholder_map(parameters: tuple[str, ...] | list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = {"device"}
    for parameter in parameters:
        key = str(parameter)
        name = _safe_identifier(key)
        candidate = name
        suffix = 2
        while candidate in used:
            candidate = f"{name}_{suffix}"
            suffix += 1
        used.add(candidate)
        mapping[key] = candidate
    return mapping


def _action_call_source(step: SkillStep, placeholder_map: dict[str, str]) -> str:
    args = [_code_literal(step.action_type)]
    kwargs: list[str] = []
    if step.target:
        kwargs.append(f"target={_template_literal(step.target, placeholder_map)}")
    if step.parameters:
        for key, value in step.parameters.items():
            if _safe_identifier(str(key)) == str(key):
                kwargs.append(f"{key}={_template_literal(value, placeholder_map)}")
            else:
                kwargs.append(f"parameters={_code_literal(step.parameters)}")
                break
    if step.valid_state is not None:
        kwargs.append(f"valid_state={_template_literal(step.valid_state, placeholder_map)}")
    if step.state_contract:
        kwargs.append(f"state_contract=C.from_dict({_code_literal(step.state_contract)})")
    if step.fixed:
        kwargs.append("fixed=True")
    if step.fixed_values:
        kwargs.append(f"fixed_values={_code_literal(step.fixed_values)}")
    return f"await action({', '.join([*args, *kwargs])})"


def _template_literal(value: Any, placeholder_map: dict[str, str]) -> str:
    if isinstance(value, str):
        parts: list[str] = []
        cursor = 0
        for match in _PLACEHOLDER_RE.finditer(value):
            if match.start() > cursor:
                parts.append(_code_literal(value[cursor : match.start()]))
            name = match.group(1)
            replacement = placeholder_map.get(name)
            if replacement is None:
                parts.append(_code_literal(match.group(0)))
            else:
                parts.append(replacement)
            cursor = match.end()
        if cursor < len(value):
            parts.append(_code_literal(value[cursor:]))
        if not parts:
            return _code_literal(value)
        if len(parts) == 1:
            return parts[0]
        return " + ".join(parts)
    return _code_literal(value)


def _code_literal(value: Any) -> str:
    return repr(value)


__all__ = [
    "CANONICAL_SKILLS_FILENAME",
    "CODE_HEADER",
    "C",
    "FlatCompileResult",
    "FlatSkillLibrary",
    "FlatSkillRepository",
    "DEFAULT_SKILLS_SOURCE_PATH",
    "DEFAULT_SKILLS_STORE_DIR",
    "R",
    "action",
    "compile_flat_skills",
    "export_skills_to_source",
    "order_skills_for_source",
    "skill",
]

"""Phase B: tag-based valid_state tiering + prompt use_skill entry guard."""

from __future__ import annotations

from guiclaw.agent import _prompt_skill_entry_allows
from guiclaw.skills.data import Skill, SkillStep
from guiclaw.skills.executor import (
    SkillExecutor,
    ValidStateMode,
    _coerce_valid_state_mode,
    _skill_is_compact,
)
from nanobot.config.schema import GuiConfig


def _skill(*steps: SkillStep, tags: tuple[str, ...] = ()) -> Skill:
    return Skill(
        skill_id="s:x:y",
        name="y",
        description="",
        app="com.x",
        platform="android",
        steps=tuple(steps),
        tags=tags,
    )


def _step(action_type="tap", target="btn", *, valid_state=None, state_contract=None,
          optional=False) -> SkillStep:
    params = {"optional": True} if optional else {}
    return SkillStep(
        action_type=action_type,
        target=target,
        parameters=params,
        valid_state=valid_state,
        state_contract=state_contract,
    )


_CONTRACT = {"anchor": {}, "signature": {"required": [{"selector": {"text": "x"}}]},
             "fingerprint": "deadbeef"}


# ---------------------------------------------------------------------------
# mode coercion (legacy bool migration)
# ---------------------------------------------------------------------------


class TestCoerceMode:
    def test_legacy_bool(self):
        assert _coerce_valid_state_mode(True) is ValidStateMode.STRICT
        # Legacy False kept deterministic contracts -> contract_only, NOT off.
        assert _coerce_valid_state_mode(False) is ValidStateMode.CONTRACT_ONLY

    def test_strings_and_none(self):
        assert _coerce_valid_state_mode("off") is ValidStateMode.OFF
        assert _coerce_valid_state_mode("contract_only") is ValidStateMode.CONTRACT_ONLY
        assert _coerce_valid_state_mode(None) is ValidStateMode.STRICT
        assert _coerce_valid_state_mode(ValidStateMode.OFF) is ValidStateMode.OFF


class TestConfigMigration:
    def test_legacy_true_maps_strict(self):
        assert GuiConfig(enable_skill_valid_state=True).skill_valid_state_mode == "strict"

    def test_legacy_false_maps_contract_only(self):
        assert GuiConfig(enable_skill_valid_state=False).skill_valid_state_mode == "contract_only"

    def test_explicit_mode_wins(self):
        cfg = GuiConfig(enable_skill_valid_state=True, skill_valid_state_mode="off")
        assert cfg.skill_valid_state_mode == "off"

    def test_default(self):
        assert GuiConfig().skill_valid_state_mode == "strict"


# ---------------------------------------------------------------------------
# _effective_mode tiering
# ---------------------------------------------------------------------------


class TestEffectiveMode:
    def _exec(self, mode):
        return SkillExecutor(backend=None, valid_state_mode=mode)

    def test_off_is_absolute(self):
        ex = self._exec("off")
        # Even an optional / visual-guarded / contracted step stays off.
        m = ex._effective_mode(_skill(), _step(state_contract=_CONTRACT, optional=True),
                               is_optional=True, visual_guarded_step=True)
        assert m is ValidStateMode.OFF

    def test_visual_guarded_forces_strict(self):
        ex = self._exec("strict")
        m = ex._effective_mode(_skill(tags=("compact_extracted",)),
                               _step(state_contract=_CONTRACT),
                               is_optional=False, visual_guarded_step=True)
        assert m is ValidStateMode.STRICT

    def test_optional_with_contract_is_contract_only(self):
        ex = self._exec("strict")
        m = ex._effective_mode(_skill(), _step(state_contract=_CONTRACT),
                               is_optional=True, visual_guarded_step=False)
        assert m is ValidStateMode.CONTRACT_ONLY

    def test_optional_without_contract_is_strict(self):
        ex = self._exec("strict")
        m = ex._effective_mode(_skill(), _step(),
                               is_optional=True, visual_guarded_step=False)
        assert m is ValidStateMode.STRICT

    def test_compact_contracted_step_downgrades(self):
        ex = self._exec("strict")
        m = ex._effective_mode(_skill(tags=("compact_extracted",)),
                               _step(state_contract=_CONTRACT),
                               is_optional=False, visual_guarded_step=False)
        assert m is ValidStateMode.CONTRACT_ONLY

    def test_compact_uncontracted_step_stays_strict(self):
        ex = self._exec("strict")
        # No contract -> must NOT run with no guard at all.
        m = ex._effective_mode(_skill(tags=("compact_extracted",)), _step(),
                               is_optional=False, visual_guarded_step=False)
        assert m is ValidStateMode.STRICT

    def test_non_compact_uses_global(self):
        ex = self._exec("strict")
        m = ex._effective_mode(_skill(), _step(state_contract=_CONTRACT),
                               is_optional=False, visual_guarded_step=False)
        assert m is ValidStateMode.STRICT

    def test_compact_helper(self):
        assert _skill_is_compact(_skill(tags=("compact", "compact_extracted")))
        assert not _skill_is_compact(_skill(tags=("compact",)))


# ---------------------------------------------------------------------------
# _validate_state mode routing
# ---------------------------------------------------------------------------


class _RaisingValidator:
    """NL validator that must never be called in non-strict modes."""

    async def validate(self, valid_state, screenshot=None):
        raise AssertionError("NL validator should not be called")

    def drain_usage(self):
        return {}


class TestValidateStateRouting:
    async def test_off_skips_contract_and_validator(self):
        ex = SkillExecutor(backend=None, state_validator=_RaisingValidator())
        # Contract present, but off skips everything.
        valid, usage, dur = await ex._validate_state(
            _step(state_contract=_CONTRACT, valid_state="x is visible"),
            None, mode=ValidStateMode.OFF,
        )
        assert valid is True and dur is None

    async def test_contract_only_skips_nl_when_no_verdict(self):
        ex = SkillExecutor(backend=None, state_validator=_RaisingValidator())
        # No contract -> no verdict -> contract_only allows without calling validator.
        valid, _u, dur = await ex._validate_state(
            _step(valid_state="x is visible"), None, mode=ValidStateMode.CONTRACT_ONLY,
        )
        assert valid is True and dur is None

    async def test_strict_calls_nl_validator(self):
        calls: list[str] = []

        class _V:
            async def validate(self, valid_state, screenshot=None):
                calls.append(valid_state)
                return False

            def drain_usage(self):
                return {}

        ex = SkillExecutor(backend=None, state_validator=_V())
        valid, _u, _d = await ex._validate_state(
            _step(valid_state="x is visible"), b"img", mode=ValidStateMode.STRICT,
        )
        assert valid is False
        assert calls == ["x is visible"]


# ---------------------------------------------------------------------------
# prompt use_skill entry guard
# ---------------------------------------------------------------------------


class _RaisingBackend:
    platform = "android"

    async def execute(self, action, timeout=30.0):
        raise RuntimeError("deeplink boom")

    async def observe(self, path, timeout=5.0):  # pragma: no cover - unused
        raise RuntimeError("no observe")


class TestOffDoesNotMaskExecutionError:
    async def test_deeplink_execution_error_fails_under_off(self):
        from guiclaw.skills.executor import ExecutionState

        step = SkillStep(
            action_type="open_deeplink",
            target="myapp://x",
            parameters={"text": "myapp://x"},
            state_contract=_CONTRACT,
        )
        skill = _skill(step)
        ex = SkillExecutor(backend=_RaisingBackend(), valid_state_mode="off")
        result = await ex.execute(skill)
        # off skips validation, but a hard backend error must not be a success.
        assert result.state is ExecutionState.FAILED


class _Obs:
    def __init__(self, texts):
        self.extra = {"ui_tree": [{"text": t} for t in texts]}


class TestEntryGuard:
    def test_open_app_entry_allowed(self):
        sk = _skill(_step("open_app", "com.x"))
        assert _prompt_skill_entry_allows(sk, None, {}) is True

    def test_deeplink_entry_allowed(self):
        sk = _skill(_step("open_deeplink", "myapp://x"))
        assert _prompt_skill_entry_allows(sk, None, {}) is True

    def test_launcher_entry_rejected(self):
        sk = _skill(_step("open_app", "com.google.android.apps.nexuslauncher"))
        assert _prompt_skill_entry_allows(sk, None, {}) is False

    def test_rendered_target_presence(self):
        sk = _skill(_step("tap", "{{q}}"))
        assert _prompt_skill_entry_allows(sk, _Obs(["Hello World"]), {"q": "Hello"}) is True
        assert _prompt_skill_entry_allows(sk, _Obs(["Hello World"]), {"q": "Nope"}) is False

    def test_no_contract_no_visible_target_rejected(self):
        sk = _skill(_step("tap", "Compose"))
        assert _prompt_skill_entry_allows(sk, _Obs(["Inbox", "Settings"]), {}) is False

    def test_empty_skill_rejected(self):
        assert _prompt_skill_entry_allows(_skill(), _Obs([]), {}) is False

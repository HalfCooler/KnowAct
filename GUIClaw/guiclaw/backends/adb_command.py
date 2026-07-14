
"""Whitelisted parameterized ADB command-skill execution."""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Awaitable, Callable

from guiclaw.action import Action

RunAdb = Callable[..., Awaitable[str]]
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)(?::([A-Za-z0-9_*]+))?\s*\}\}")
_SAFE_INT_LIST_RE = re.compile(r"^[0-9]+(?:,[0-9]+)*$")
_OPTIONAL_VERIFIER_SKILLS = frozenset({"clock.set_alarm_intent"})
_DAY_NAME_TO_INT = {
    "sunday": 1,
    "sun": 1,
    "monday": 2,
    "mon": 2,
    "tuesday": 3,
    "tue": 3,
    "tues": 3,
    "wednesday": 4,
    "wed": 4,
    "thursday": 5,
    "thu": 5,
    "thur": 5,
    "thurs": 5,
    "friday": 6,
    "fri": 6,
    "saturday": 7,
    "sat": 7,
}
_DAY_ALIAS_TO_CSV = {
    "": "",
    "none": "",
    "once": "",
    "one-time": "",
    "one time": "",
    "no repeat": "",
    "weekend": "1,7",
    "weekends": "1,7",
    "sat,sun": "1,7",
    "saturday,sunday": "1,7",
    "weekday": "2,3,4,5,6",
    "weekdays": "2,3,4,5,6",
    "workday": "2,3,4,5,6",
    "workdays": "2,3,4,5,6",
    "everyday": "1,2,3,4,5,6,7",
    "every day": "1,2,3,4,5,6,7",
    "daily": "1,2,3,4,5,6,7",
    "all": "1,2,3,4,5,6,7",
}


class AdbCommandError(RuntimeError):
    """Raised when a whitelisted adb command skill fails."""


class AdbCommandRunner:
    """Execute registry-defined ADB command skills without accepting raw shell."""

    def __init__(self, run_adb: RunAdb, *, registry_path: str | Path | None = None) -> None:
        self._run_adb = run_adb
        self._registry_path = Path(registry_path).expanduser() if registry_path else None
        self._registry: dict[str, dict[str, Any]] | None = None

    async def execute(self, action: Action, *, timeout: float) -> str:
        command_id = (action.command_id or action.text or "").strip()
        if not command_id:
            raise ValueError("adb_command action requires command_id or text")
        skill = self._load_registry().get(command_id)
        if skill is None:
            raise ValueError(f"Unknown adb command skill: {command_id}")
        params = self._prepare_params(skill, action)
        executor = skill.get("executor") or {}
        output = await self._execute_executor(executor, params, timeout=timeout)
        verifier = skill.get("verifier")
        if skill.get("kind") == "write" and verifier:
            try:
                await self._verify(verifier, params, timeout=timeout)
            except Exception as exc:
                if not _verifier_warn_only(skill):
                    raise
                output = (
                    f"{output}\noptional verifier failed: {exc}"
                    if output
                    else f"optional verifier failed: {exc}"
                )
        return f"adb_command {command_id} succeeded" + (f"\n{output}" if output else "")

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        if self._registry is not None:
            return self._registry
        path = self._registry_path or _find_registry_path()
        if path is None:
            raise FileNotFoundError(
                "ADB command skill registry not found. Set GUICLAW_ADB_COMMAND_SKILLS_PATH."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        self._registry = {str(item["id"]): item for item in data.get("skills", [])}
        return self._registry

    def _prepare_params(self, skill: dict[str, Any], action: Action) -> dict[str, Any]:
        raw: dict[str, Any] = dict(action.params or {})
        for key, value in action.extras:
            raw.setdefault(str(key), value)
        schema = skill.get("params") or {}
        params: dict[str, Any] = {}
        for name, spec in schema.items():
            if name in raw:
                params[name] = _coerce_param(name, raw[name], spec)
            elif "default" in spec:
                params[name] = spec["default"]
            elif spec.get("required"):
                raise ValueError(f"adb command {skill.get('id')} missing required param {name!r}")
        for name, value in raw.items():
            if name not in params:
                params[name] = value
        _add_derived_params(params)
        _validate_param_constraints(params, schema)
        return params

    async def _execute_executor(self, executor: dict[str, Any], params: dict[str, Any], *, timeout: float) -> str:
        kind = executor.get("type")
        if kind == "adb_template":
            if "templates" in executor:
                enabled = bool(params.get("enabled"))
                template = executor["templates"]["enabled_true" if enabled else "enabled_false"]
            else:
                template = str(executor.get("template") or "")
            return await self._run_adb_template(template, params, timeout=timeout, root=bool(executor.get("root_required")))
        if kind == "adb_sequence":
            if executor.get("root_required"):
                await self._try_root(timeout=timeout)
            outputs: list[str] = []
            for step in executor.get("steps") or []:
                if not _condition_passes(str(step.get("condition") or ""), params):
                    continue
                outputs.append(await self._run_adb_template(str(step.get("template") or ""), params, timeout=timeout))
            return "\n".join(part for part in outputs if part)
        if kind == "adb_sqlite":
            sql = _render_template(str(executor.get("sql_template") or ""), params)
            db_path = str(executor.get("db_path") or "")
            return await self._run_sqlite(db_path, sql, timeout=timeout)
        if kind == "python_helper":
            return _run_python_helper(str(executor.get("helper") or ""), params)
        raise ValueError(f"Unsupported adb command executor type: {kind!r}")

    async def _verify(self, verifier: dict[str, Any], params: dict[str, Any], *, timeout: float) -> None:
        kind = verifier.get("type")
        if kind == "skill_ref":
            ref = str(verifier.get("skill_id") or "")
            if not ref:
                raise ValueError("skill_ref verifier missing skill_id")
            verify_params = dict(params)
            if ref == "files.unzip_list" and "archive_path" not in verify_params:
                directory = str(verify_params.get("directory") or "").rstrip("/")
                archive = str(verify_params.get("archive_name") or "")
                verify_params["archive_path"] = f"{directory}/{archive}"
            if ref == "media.query_image_path" and "image_name" not in verify_params:
                verify_params["image_name"] = Path(str(verify_params.get("path") or "")).name
            output = await self._execute_executor(self._load_registry()[ref].get("executor") or {}, verify_params, timeout=timeout)
            if not output.strip():
                raise AdbCommandError(f"adb command verifier {ref} returned empty output")
            return
        if kind == "adb_template":
            output = await self._run_adb_template(str(verifier.get("template") or ""), params, timeout=timeout)
            success = str(verifier.get("success") or "").lower()
            if "stdout is empty" in success and output.strip():
                raise AdbCommandError(f"adb command verifier expected empty stdout, got: {output}")
            if "equals '1'" in success:
                expected = "1" if bool(params.get("enabled")) else "0"
                if output.strip() != expected:
                    raise AdbCommandError(f"adb command verifier expected {expected}, got: {output}")
            return

    async def _run_adb_template(self, template: str, params: dict[str, Any], *, timeout: float, root: bool = False) -> str:
        if not template.strip():
            raise ValueError("empty adb command template")
        if root:
            await self._try_root(timeout=timeout)
        return await self._run_adb_string(_render_template(template, params), timeout=timeout)

    async def _run_adb_string(self, command: str, *, timeout: float) -> str:
        parts = shlex.split(command)
        if not parts:
            raise ValueError("empty adb command")
        if parts[0] == "adb":
            parts = parts[1:]
        if not parts:
            raise ValueError("adb command has no arguments")
        return await self._run_adb(*parts, timeout=timeout)

    async def _run_sqlite(self, db_path: str, sql: str, *, timeout: float) -> str:
        if not db_path or not sql:
            raise ValueError("adb_sqlite requires db_path and sql")
        commands = [
            f"adb shell su 0 sqlite3 {shlex.quote(db_path)} {shlex.quote(sql)}",
            f"adb shell su root sqlite3 {shlex.quote(db_path)} {shlex.quote(sql)}",
        ]
        errors: list[str] = []
        for command in commands:
            try:
                output = await self._run_adb_string(command, timeout=timeout)
            except Exception as exc:
                errors.append(str(exc))
                continue
            if "error" not in output.lower():
                return output
        raise AdbCommandError("adb sqlite command failed: " + " | ".join(errors))

    async def _try_root(self, *, timeout: float) -> None:
        try:
            await self._run_adb("root", timeout=timeout)
        except Exception:
            # Some images keep shell root via su even when `adb root` is noisy.
            pass


def _find_registry_path() -> Path | None:
    env_path = os.getenv("GUICLAW_ADB_COMMAND_SKILLS_PATH", "").strip()
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path
    candidates: list[Path] = []
    cwd = Path.cwd()
    for root in (cwd, *cwd.parents):
        candidates.append(root / "configs" / "adb_command_skills.mobileworld.json")
    for key in ("GUI_CLAW_PATH", "GUICLAW_WORKSPACE", "NANOBOT_WORKSPACE"):
        value = os.getenv(key, "").strip()
        if value:
            candidates.append(Path(value).expanduser() / "configs" / "adb_command_skills.mobileworld.json")
    for path in candidates:
        if path.exists():
            return path
    return None


def _coerce_param(name: str, value: Any, spec: dict[str, Any]) -> Any:
    typ = spec.get("type")
    if typ == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        raise ValueError(f"param {name!r} must be boolean")
    if typ == "integer":
        return int(value)
    if typ == "array":
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, (list, tuple)):
            return list(value)
        raise ValueError(f"param {name!r} must be array")
    if typ in {"string", "path", "filename", "string_or_integer"}:
        return value if typ == "string_or_integer" and isinstance(value, int) else str(value)
    return value


def _add_derived_params(params: dict[str, Any]) -> None:
    days_csv = _normalize_days_csv(params.get("days_csv"))
    params["days_csv"] = days_csv
    if days_csv:
        params["days_arg"] = f"--eia android.intent.extra.alarm.DAYS {days_csv}"
    else:
        params["days_arg"] = ""
    if params.get("hour") not in (None, "") and params.get("minute") not in (None, ""):
        params["where_hour_minute"] = f"WHERE hour={int(params['hour'])} AND minutes={int(params['minute'])}"
    else:
        params["where_hour_minute"] = ""
    sms_contains = str(params.get("contains") or "").strip()
    params["sms_where_arg"] = (
        "--where " + shlex.quote(f"body LIKE '%{_escape_sql_like(sms_contains)}%'")
        if sms_contains
        else ""
    )
    contact_clauses: list[str] = []
    name_contains = str(params.get("name_contains") or "").strip()
    phone_number = str(params.get("phone_number") or "").strip()
    if name_contains:
        contact_clauses.append(f"display_name LIKE '%{_escape_sql_like(name_contains)}%'")
    if phone_number:
        contact_clauses.append(f"data1 LIKE '%{_escape_sql_like(phone_number)}%'")
    params["contacts_where_arg"] = (
        "--where " + shlex.quote(" AND ".join(contact_clauses)) if contact_clauses else ""
    )
    image_name = str(params.get("image_name") or "").strip()
    if image_name:
        params["image_glob"] = f"*{image_name.replace(chr(0), '')}*"
    else:
        params["image_glob"] = "*"


def _validate_param_constraints(params: dict[str, Any], schema: dict[str, Any]) -> None:
    for name, spec in schema.items():
        if name not in params:
            continue
        value = params[name]
        if spec.get("type") == "integer":
            if "minimum" in spec and int(value) < int(spec["minimum"]):
                raise ValueError(f"param {name!r} below minimum")
            if "maximum" in spec and int(value) > int(spec["maximum"]):
                raise ValueError(f"param {name!r} above maximum")
        if spec.get("type") == "filename":
            text = str(value)
            if not text or "/" in text or "\x00" in text or text in {".", ".."}:
                raise ValueError(f"unsafe filename param {name!r}: {text!r}")
        prefixes = spec.get("allowed_prefixes") or []
        if prefixes:
            values = value if isinstance(value, list) else [value]
            for item in values:
                text = str(item)
                if "\x00" in text or "/../" in text or text.endswith("/.."):
                    raise ValueError(f"unsafe path param {name!r}: {text!r}")
                if not any(text == prefix or text.startswith(str(prefix).rstrip("/") + "/") for prefix in prefixes):
                    raise ValueError(f"path param {name!r} outside allowed prefixes: {text!r}")


def _render_template(template: str, params: dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        name, fmt = match.group(1), match.group(2) or ""
        if name not in params:
            raise ValueError(f"template references missing param {name!r}")
        value = params[name]
        if fmt == "q":
            return shlex.quote(str(value))
        if fmt == "q*":
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"template param {name!r} must be a list for q*")
            return " ".join(shlex.quote(str(item)) for item in value)
        if fmt == "bool":
            return "true" if bool(value) else "false"
        if fmt == "intbool":
            return "1" if bool(value) else "0"
        return str(value)
    return _PLACEHOLDER_RE.sub(repl, template)


def _normalize_days_csv(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("，", ",").replace(";", ",").replace("|", ",")
    text = re.sub(r"\s+", " ", text)
    if text in _DAY_ALIAS_TO_CSV:
        return _DAY_ALIAS_TO_CSV[text]
    compact = text.replace(" ", "")
    if compact in _DAY_ALIAS_TO_CSV:
        return _DAY_ALIAS_TO_CSV[compact]
    if _SAFE_INT_LIST_RE.match(compact):
        values = [int(part) for part in compact.split(",") if part]
        invalid = [value for value in values if value < 1 or value > 7]
        if invalid:
            raise ValueError("days_csv values must be Android Calendar day integers 1..7")
        return ",".join(str(value) for value in values)
    tokens = [token for token in re.split(r"[\s,/+]+", text) if token and token != "and"]
    if tokens:
        values: list[int] = []
        for token in tokens:
            if token not in _DAY_NAME_TO_INT:
                raise ValueError(
                    "days_csv must be comma-separated Android day integers, "
                    "or a supported alias such as weekend/weekdays/everyday"
                )
            values.append(_DAY_NAME_TO_INT[token])
        deduped = sorted(set(values))
        return ",".join(str(value) for value in deduped)
    return ""


def _escape_sql_like(value: str) -> str:
    return value.replace("'", "''").replace("\x00", "")


def _verifier_warn_only(skill: dict[str, Any]) -> bool:
    if bool(skill.get("verifier_warn_only") or skill.get("optional_verifier")):
        return True
    return str(skill.get("id") or "") in _OPTIONAL_VERIFIER_SKILLS


def _json_default(value: Any) -> str:
    return str(value)


def _run_python_helper(helper: str, params: dict[str, Any]) -> str:
    if helper == "mobileworld.calendar.insert_event":
        from mobile_world.runtime.app_helpers import fossify_calendar

        title = str(params.get("title") or "").strip()
        start_time = str(params.get("start_time") or "").strip()
        end_time = str(params.get("end_time") or "").strip()
        if not title or not start_time or not end_time:
            raise ValueError("calendar.insert_event requires title, start_time, and end_time")
        reminder_minutes = int(params.get("reminder_minutes", -1))
        fossify_calendar.insert_calendar_event(
            title=title,
            start_time=start_time,
            end_time=end_time,
            location=str(params.get("location") or ""),
            description=str(params.get("description") or ""),
            reminder_1_minutes=reminder_minutes,
        )
        events = fossify_calendar.get_calendar_events(
            time_range=[start_time, end_time],
            format_timestamp=True,
        )
        return json.dumps({"created": True, "events": events}, ensure_ascii=False, default=_json_default)

    if helper == "mobileworld.mattermost.send_message":
        from mobile_world.runtime.app_helpers import mattermost

        username = str(params.get("username") or "harry.kong@neuralforge.ai")
        password = str(params.get("password") or mattermost.DEFAULT_PASSWORD)
        team = str(params.get("team") or mattermost.TEAM_NAME)
        channel = str(params.get("channel") or "").strip()
        message = str(params.get("message") or "")
        reply_to = str(params.get("reply_to") or "").strip() or None
        if not channel or not message:
            raise ValueError("mattermost.send_message requires channel and message")
        cli = mattermost.MattermostCLI()
        if not cli.login(username, password):
            raise RuntimeError(f"failed to login to Mattermost as {username}")
        try:
            if not cli.send_message(team=team, channel=channel, message=message, reply_to=reply_to):
                raise RuntimeError("failed to send Mattermost message")
        finally:
            cli.logout()
        latest_messages = mattermost.get_latest_messages() or []
        latest = latest_messages[:3]
        return json.dumps({"sent": True, "latest_messages": latest}, ensure_ascii=False, default=_json_default)

    if helper == "mobileworld.mattermost.create_channel":
        from mobile_world.runtime.app_helpers import mattermost

        username = str(params.get("username") or "harry.kong@neuralforge.ai")
        password = str(params.get("password") or mattermost.DEFAULT_PASSWORD)
        team = str(params.get("team") or mattermost.TEAM_NAME)
        channel = str(params.get("channel") or "").strip()
        display_name = str(params.get("display_name") or channel).strip()
        private = bool(params.get("private"))
        purpose = str(params.get("purpose") or "")
        header = str(params.get("header") or "")
        if not channel:
            raise ValueError("mattermost.create_channel requires channel")
        cli = mattermost.MattermostCLI()
        if not cli.login(username, password):
            raise RuntimeError(f"failed to login to Mattermost as {username}")
        try:
            if not cli.create_channel(
                team=team,
                channel_name=channel,
                display_name=display_name,
                private=private,
                purpose=purpose,
                header=header,
            ):
                raise RuntimeError("failed to create Mattermost channel")
        finally:
            cli.logout()
        channel_info = mattermost.get_channel_info(channel_name=channel)
        return json.dumps(
            {"created": True, "channel": channel, "channel_info": channel_info},
            ensure_ascii=False,
            default=_json_default,
        )

    if helper == "mobileworld.mattermost.add_users":
        from mobile_world.runtime.app_helpers import mattermost

        username = str(params.get("username") or "harry.kong@neuralforge.ai")
        password = str(params.get("password") or mattermost.DEFAULT_PASSWORD)
        team = str(params.get("team") or mattermost.TEAM_NAME)
        channel = str(params.get("channel") or "").strip()
        users = params.get("users")
        if isinstance(users, str):
            if users.strip().lower() in {"everyone", "all", "all users", "all members"}:
                connection, cursor = mattermost.connect_to_postgres()
                if connection is None or cursor is None:
                    raise RuntimeError("failed to connect to Mattermost database")
                try:
                    cursor.execute(
                        "SELECT email FROM users WHERE deleteat = 0 AND email <> '' "
                        "ORDER BY createat ASC"
                    )
                    users = [row[0] for row in cursor.fetchall() if row and row[0]]
                finally:
                    cursor.close()
                    connection.close()
            else:
                users = [part.strip() for part in users.split(",") if part.strip()]
        if not channel or not users:
            raise ValueError("mattermost.add_users requires channel and users")
        cli = mattermost.MattermostCLI()
        if not cli.login(username, password):
            raise RuntimeError(f"failed to login to Mattermost as {username}")
        try:
            if not cli.add_users_to_channel(team=team, channel=channel, users=list(users)):
                raise RuntimeError("failed to add Mattermost users")
        finally:
            cli.logout()
        return json.dumps({"added": True, "channel": channel, "users": users}, ensure_ascii=False)

    if helper == "mobileworld.mastodon.query_latest_toots":
        from mobile_world.runtime.app_helpers import mastodon

        username = str(params.get("username") or "").strip()
        limit = int(params.get("limit", 5))
        if not username:
            raise ValueError("mastodon.query_latest_toots requires username")
        toots = mastodon.get_latest_toots_by_username(username, limit=limit)
        return json.dumps({"username": username, "toots": toots}, ensure_ascii=False, default=_json_default)

    if helper == "mobileworld.mastodon.query_profile":
        from mobile_world.runtime.app_helpers import mastodon

        username = str(params.get("username") or "").strip()
        if not username:
            raise ValueError("mastodon.query_profile requires username")
        profile = mastodon.get_user_account_info(username)
        return json.dumps({"username": username, "profile": profile}, ensure_ascii=False, default=_json_default)

    raise ValueError(f"unsupported python_helper: {helper!r}")


def _condition_passes(condition: str, params: dict[str, Any]) -> bool:
    if not condition:
        return True
    normalized = condition.strip().lower()
    if normalized == "delete_sources == true":
        return bool(params.get("delete_sources"))
    raise ValueError(f"unsupported adb command condition: {condition}")

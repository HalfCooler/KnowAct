"""
guiclaw.backends.keycodes
==========================
Shared key-name definitions and platform-specific mappings for device backends.
"""

from __future__ import annotations

from enum import StrEnum


class CommonKey(StrEnum):
    """Canonical key names recognised across platforms."""

    HOME = "home"
    BACK = "back"
    ENTER = "enter"
    RETURN = "return"
    TAB = "tab"
    DELETE = "delete"
    BACKSPACE = "backspace"
    VOLUME_UP = "volumeup"
    VOLUME_DOWN = "volumedown"
    POWER = "power"
    MENU = "menu"
    APP_SWITCH = "app_switch"
    RECENTS = "recents"
    ESCAPE = "escape"
    SPACE = "space"
    SEARCH = "search"
    CAMERA = "camera"
    DPAD_LEFT = "left"
    DPAD_RIGHT = "right"
    DPAD_UP = "up"
    DPAD_DOWN = "down"
# ---------------------------------------------------------------------------
# Cross-platform aliases — canonical name → (canonical_name,)
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {
    "return": CommonKey.ENTER,
    "backspace": CommonKey.DELETE,
    "recents": CommonKey.APP_SWITCH,
    "volume_up": CommonKey.VOLUME_UP,
    "volume_down": CommonKey.VOLUME_DOWN,
}


def canonical_key_name(key: str) -> str:
    """Return the canonical ``CommonKey`` name for *key*, or *key* unchanged."""
    return _ALIASES.get(key.strip().lower(), key.strip().lower())


# ---------------------------------------------------------------------------
# Pre-built platform maps (canonical name → platform code)
# ---------------------------------------------------------------------------

ANDROID_KEYCODE_MAP: dict[str, str] = {
    CommonKey.HOME: "KEYCODE_HOME",
    CommonKey.BACK: "KEYCODE_BACK",
    CommonKey.ENTER: "KEYCODE_ENTER",
    CommonKey.TAB: "KEYCODE_TAB",
    CommonKey.DELETE: "KEYCODE_DEL",
    CommonKey.VOLUME_UP: "KEYCODE_VOLUME_UP",
    CommonKey.VOLUME_DOWN: "KEYCODE_VOLUME_DOWN",
    CommonKey.POWER: "KEYCODE_POWER",
    CommonKey.MENU: "KEYCODE_MENU",
    CommonKey.APP_SWITCH: "KEYCODE_APP_SWITCH",
    CommonKey.ESCAPE: "KEYCODE_ESCAPE",
    CommonKey.SPACE: "KEYCODE_SPACE",
    CommonKey.SEARCH: "KEYCODE_SEARCH",
    CommonKey.CAMERA: "KEYCODE_CAMERA",
    CommonKey.DPAD_LEFT: "KEYCODE_DPAD_LEFT",
    CommonKey.DPAD_RIGHT: "KEYCODE_DPAD_RIGHT",
    CommonKey.DPAD_UP: "KEYCODE_DPAD_UP",
    CommonKey.DPAD_DOWN: "KEYCODE_DPAD_DOWN",
}

HDC_KEYCODE_MAP: dict[str, str] = {
    CommonKey.HOME: "Home",
    CommonKey.BACK: "Back",
    CommonKey.ENTER: "2054",
    CommonKey.DELETE: "2055",
    CommonKey.VOLUME_UP: "2072",
    CommonKey.VOLUME_DOWN: "2073",
    CommonKey.POWER: "2050",
}

IOS_KEYCODE_MAP: dict[str, str] = {
    CommonKey.HOME: "home",
    CommonKey.VOLUME_UP: "volumeUp",
    CommonKey.VOLUME_DOWN: "volumeDown",
}

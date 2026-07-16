"""GUIClaw skill extraction, retrieval, and execution submodules."""

_DESKTOP_PLATFORMS = frozenset({"linux", "macos", "windows"})


def skills_supported_for_platform(platform: str) -> bool:
    """Return whether stable skill execution is supported on the platform."""

    return str(platform).strip().lower() not in _DESKTOP_PLATFORMS

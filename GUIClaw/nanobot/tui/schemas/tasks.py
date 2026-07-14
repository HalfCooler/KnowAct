"""Task capability and launch schemas for the TUI backend."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class TaskContractResponse(BaseModel):
    """Read-only task capability payload."""

    name: str
    mutable: bool
    phase: int
    status: str


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NanobotLaunchOptions(_StrictModel):
    require_background_isolation: bool = False
    acknowledge_background_fallback: bool = False


class NanobotOpenUrlLaunchRequest(NanobotLaunchOptions):
    kind: Literal["nanobot_open_url"]
    url: HttpUrl


class NanobotOpenSettingsLaunchRequest(NanobotLaunchOptions):
    kind: Literal["nanobot_open_settings"]
    panel: Literal["network", "display", "privacy", "bluetooth"]


class GUIClawLaunchAppRequest(_StrictModel):
    kind: Literal["guiclaw_launch_app"]
    app_id: Literal["calculator", "notepad", "settings", "terminal"]
    backend: Literal["dry-run", "local"] | None = None


class GUIClawOpenSettingsRequest(_StrictModel):
    kind: Literal["guiclaw_open_settings"]
    panel: Literal["network", "display", "privacy", "bluetooth"]
    backend: Literal["dry-run", "local"] | None = None


TaskLaunchRequest = Annotated[
    NanobotOpenUrlLaunchRequest
    | NanobotOpenSettingsLaunchRequest
    | GUIClawLaunchAppRequest
    | GUIClawOpenSettingsRequest,
    Field(discriminator="kind"),
]


class LaunchRunResponse(_StrictModel):
    run_id: str
    status: str
    accepted_at: str

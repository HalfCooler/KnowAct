"""Helpers for serving the built web workspace from the TUI FastAPI app."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles


def frontend_dist_path() -> Path:
    """Return the packaged dist directory for the TUI web frontend."""

    return Path(str(resources.files("nanobot.tui.web").joinpath("dist")))


def frontend_index_path() -> Path:
    """Return the built SPA shell entrypoint."""

    return frontend_dist_path() / "index.html"


def frontend_assets_path() -> Path:
    """Return the built static assets directory."""

    return frontend_dist_path() / "assets"


def frontend_assets_ready() -> bool:
    """Report whether a built frontend bundle is available."""

    return frontend_index_path().is_file()


def frontend_startup_guidance() -> str:
    """Return deterministic guidance when the frontend bundle is missing."""

    return """
<html>
  <body>
    <h1>Nanobot web frontend build missing</h1>
    <p>Build the frontend bundle before serving the packaged shell.</p>
    <ul>
      <li><code>npm --prefix nanobot/tui/web run build</code></li>
      <li><code>npm --prefix nanobot/tui/web run dev</code></li>
    </ul>
  </body>
</html>
""".strip()


def frontend_shell_response() -> Response:
    """Return the built SPA shell or operator guidance when assets are missing."""

    index_path = frontend_index_path()
    if not index_path.is_file():
        return HTMLResponse(frontend_startup_guidance(), status_code=503)
    return FileResponse(index_path)


def install_frontend_routes(app: FastAPI) -> None:
    """Mount built assets and SPA deep-link fallbacks onto the app."""

    assets_path = frontend_assets_path()
    if assets_path.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_path)), name="tui-web-assets")

    def serve_shell() -> Response:
        return frontend_shell_response()

    app.add_api_route("/", serve_shell, include_in_schema=False, methods=["GET"])
    app.add_api_route("/chat", serve_shell, include_in_schema=False, methods=["GET"])
    app.add_api_route("/chat/{path:path}", serve_shell, include_in_schema=False, methods=["GET"])
    app.add_api_route("/operations", serve_shell, include_in_schema=False, methods=["GET"])
    app.add_api_route(
        "/operations/{path:path}",
        serve_shell,
        include_in_schema=False,
        methods=["GET"],
    )

"""MobileWorld helper compatibility functions used by vendored agents."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image


def pil_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def pil_adaptive_resize(image: Image.Image, max_dimension: int) -> tuple[Image.Image, float, float]:
    width, height = image.size
    largest = max(width, height)
    if largest <= max_dimension:
        return image, 1.0, 1.0
    scale = max_dimension / largest
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))))
    return resized, scale, scale


def reverse_swipe_direction(direction: str) -> str:
    return {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left",
    }.get(str(direction).lower(), str(direction).lower())


def add_period_robustly(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return text
    return text if text[-1] in ".!?。！？" else text + "."


def mask_api_key(api_key: str | None) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "***"
    return api_key[:4] + "***" + api_key[-4:]


def pretty_print_messages(messages: list[dict[str, Any]], max_messages: int = 10) -> None:
    del max_messages
    # GUIClaw tracing handles prompt snapshots; this compatibility hook is quiet.
    return None

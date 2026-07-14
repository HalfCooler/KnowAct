"""
guiclaw.image_utils
===================
Shared image-scaling helpers used by both the main agent loop and the skill
executor's grounding/validation sub-loops.

These live at the ``guiclaw`` root so neither ``agent.py`` nor
``skills/executor.py`` needs to import private helpers from the other.
"""

from __future__ import annotations

import io


def normalize_image_scale_ratio(scale_ratio: float | None) -> float:
    """Normalize user-provided image scaling ratio to a safe range."""
    if scale_ratio is None:
        return 0.5
    try:
        value = float(scale_ratio)
    except (TypeError, ValueError):
        return 0.5
    if value <= 0:
        return 0.5
    return min(1.0, value)


def scale_image(data: bytes, *, scale_ratio: float = 0.5) -> bytes:
    """Return *data* scaled by *scale_ratio* as PNG bytes.

    Falls back to the original bytes if PIL is unavailable or the image cannot
    be decoded (e.g. non-PNG/JPEG formats the LLM provider may still accept).
    """
    scale_ratio = normalize_image_scale_ratio(scale_ratio)
    if scale_ratio >= 1.0:
        return data
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as img:
            w, h = img.size
            scaled = img.resize(
                (max(1, int(w * scale_ratio)), max(1, int(h * scale_ratio))),
                Image.LANCZOS,
            )
            buf = io.BytesIO()
            scaled.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        return data

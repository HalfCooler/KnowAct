"""Phase 5 adapter documentation contract tests."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTERS_DOC = REPO_ROOT / "ADAPTERS.md"
INTERFACES_FILE = REPO_ROOT / "guiclaw" / "interfaces.py"
POINTER_SENTENCE = (
    "For host-agent adapter examples, see repo-root ADAPTERS.md and "
    "nanobot/agent/gui_adapter.py."
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_adapters_doc_contains_required_sections() -> None:
    text = _read_text(ADAPTERS_DOC)

    assert "# GUIClaw Adapter Patterns" in text
    assert "## LLMProvider" in text
    assert "## DeviceBackend" in text
    assert "class ExampleHostLLMAdapter" in text
    assert "NanobotLLMAdapter" in text
    assert "nanobot/agent/gui_adapter.py" in text


def test_adapter_pointer_exists_in_code() -> None:
    text = _read_text(INTERFACES_FILE)

    assert POINTER_SENTENCE in text

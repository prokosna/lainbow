"""The committed configuration reference must match the settings classes."""

from pathlib import Path

from lainbow.core.settings_doc import render_markdown

DOC = Path(__file__).resolve().parents[2] / "docs" / "configuration.md"


def test_configuration_doc_is_up_to_date() -> None:
    assert DOC.read_text(encoding="utf-8") == render_markdown(), (
        "docs/configuration.md is out of date: run scripts/gen-docs.sh"
    )

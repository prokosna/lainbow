"""Tests for library walking and tag reading (no database)."""

from __future__ import annotations

from pathlib import Path

from lainbow.server.library import read_tags, walk_library


def test_walk_library(tmp_path: Path) -> None:
    for rel in ["b/z.FLAC", "a/y.mp3", "a/note.txt", ".hidden/x.flac", "a/.h.flac", "top.Opus"]:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
    (tmp_path / "link").symlink_to(tmp_path / "a", target_is_directory=True)

    files = list(walk_library(tmp_path))
    assert [f.path for f in files] == ["a/y.mp3", "b/z.FLAC", "top.Opus"]
    assert all(f.size_bytes == 1 and f.mtime.tzinfo is not None for f in files)


def test_read_tags_garbage(tmp_path: Path) -> None:
    f = tmp_path / "bad.flac"
    f.write_bytes(b"not audio at all")
    tags = read_tags(f)
    assert tags.title is None
    assert tags.duration_sec is None

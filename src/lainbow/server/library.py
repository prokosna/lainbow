"""Library scanning: keep the `tracks` table in sync with the music folder."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import mutagen
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session, sessionmaker

from lainbow.db.schema import Track
from lainbow.db.tracks import forget_analysis

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".flac", ".wv", ".mp3", ".wav", ".ogg", ".opus", ".m4a", ".mp4", ".aiff", ".aif"}
)

_LOG_EVERY = 5000


class LibraryUnavailableError(RuntimeError):
    """The music folder looks empty or unmounted while tracks are known."""


@dataclass(frozen=True)
class FileInfo:
    """An audio file found on disk."""

    path: str
    mtime: datetime
    size_bytes: int


def _walk(root: Path, rel: str) -> Iterator[FileInfo]:
    directory = root / rel if rel else root
    try:
        with os.scandir(directory) as it:
            entries = sorted(it, key=lambda e: e.name)
    except OSError as e:
        logger.warning("cannot read directory %s: %s", directory, e)
        return
    for entry in entries:
        if entry.name.startswith("."):
            continue
        child = f"{rel}/{entry.name}" if rel else entry.name
        try:
            if entry.is_dir(follow_symlinks=False):
                yield from _walk(root, child)
            elif entry.is_file() and os.path.splitext(entry.name)[1].lower() in AUDIO_EXTENSIONS:
                st = entry.stat()
                yield FileInfo(
                    path=child,
                    mtime=datetime.fromtimestamp(st.st_mtime, tz=UTC),
                    size_bytes=st.st_size,
                )
        except OSError as e:
            logger.warning("cannot stat %s: %s", child, e)


def walk_library(root: Path) -> Iterator[FileInfo]:
    """Yield audio files under root in sorted path order, skipping hidden and symlinked dirs."""
    yield from _walk(root, "")


@dataclass(frozen=True)
class Tags:
    """Tags and duration read from an audio file."""

    title: str | None = None
    artist: str | None = None
    albumartist: str | None = None
    album: str | None = None
    genre: str | None = None
    duration_sec: float | None = None


def _first(tags: object, key: str) -> str | None:
    try:
        values = tags[key]  # type: ignore[index]  # mutagen easy tags are untyped mappings
    except (KeyError, TypeError):
        return None
    if isinstance(values, str):
        values = [values]
    for v in values:
        s = str(v).strip()
        return s or None
    return None


def read_tags(path: Path) -> Tags:
    """Read tags with mutagen. Unreadable files give all-None tags."""
    try:
        f = mutagen.File(path, easy=True)
        if f is None:
            return Tags()
        tags = f.tags
        duration = getattr(f.info, "length", None)
        return Tags(
            title=_first(tags, "title") if tags is not None else None,
            artist=_first(tags, "artist") if tags is not None else None,
            albumartist=_first(tags, "albumartist") if tags is not None else None,
            album=_first(tags, "album") if tags is not None else None,
            genre=_first(tags, "genre") if tags is not None else None,
            duration_sec=float(duration) if duration is not None else None,
        )
    except Exception:
        return Tags()


@dataclass(frozen=True)
class ScanResult:
    """Counts of what a scan did."""

    added: int
    updated: int
    removed: int
    unchanged: int
    unreadable: int


def scan(
    session_factory: sessionmaker[Session],
    root: Path,
    *,
    read_tags: Callable[[Path], Tags] = read_tags,
    batch_size: int = 500,
) -> ScanResult:
    """Synchronize `tracks` with the files under root."""
    start = datetime.now(UTC)
    added = updated = unchanged = unreadable = 0
    seen_files = 0

    with session_factory() as session:
        known = {
            path: (tid, mtime, size)
            for path, tid, mtime, size in session.execute(
                select(Track.path, Track.id, Track.mtime, Track.size_bytes)
            )
        }
        unchanged_ids: list[int] = []
        pending = 0

        def flush() -> None:
            nonlocal pending
            if unchanged_ids:
                session.execute(
                    update(Track).where(Track.id.in_(unchanged_ids)).values(seen_at=start)
                )
                unchanged_ids.clear()
            session.commit()
            pending = 0

        for info in walk_library(root):
            seen_files += 1
            if seen_files % _LOG_EVERY == 0:
                logger.info("scan: %d files visited", seen_files)
            row = known.get(info.path)
            if row is not None and row[1] == info.mtime and row[2] == info.size_bytes:
                unchanged += 1
                unchanged_ids.append(row[0])
            else:
                tags = read_tags(root / info.path)
                if tags.duration_sec is None or tags.duration_sec <= 0:
                    unreadable += 1
                    if row is not None:
                        # Keep the existing row alive; its file is still there.
                        unchanged_ids.append(row[0])
                else:
                    values = {
                        "title": tags.title,
                        "artist": tags.artist,
                        "albumartist": tags.albumartist,
                        "album": tags.album,
                        "genre": tags.genre,
                        "duration_sec": tags.duration_sec,
                        "mtime": info.mtime,
                        "size_bytes": info.size_bytes,
                        "seen_at": start,
                    }
                    if row is None:
                        session.add(Track(path=info.path, **values))
                        added += 1
                    else:
                        session.execute(update(Track).where(Track.id == row[0]).values(**values))
                        forget_analysis(session, row[0])
                        updated += 1
            pending += 1
            if pending >= batch_size:
                flush()
        flush()

        if seen_files == 0 and known:
            raise LibraryUnavailableError(
                f"no audio files found under {root} but {len(known)} tracks are known; "
                "is the music folder mounted? Nothing was deleted."
            )

        result = session.execute(delete(Track).where(Track.seen_at < start).returning(Track.id))
        removed = len(result.all())
        session.commit()

    scan_result = ScanResult(added, updated, removed, unchanged, unreadable)
    logger.info(
        "scan finished: %d added, %d updated, %d removed, %d unchanged, %d unreadable",
        added,
        updated,
        removed,
        unchanged,
        unreadable,
    )
    return scan_result


__all__ = [
    "AUDIO_EXTENSIONS",
    "FileInfo",
    "LibraryUnavailableError",
    "ScanResult",
    "Tags",
    "read_tags",
    "scan",
    "walk_library",
]

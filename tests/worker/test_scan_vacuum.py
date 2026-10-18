import datetime
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from sqlmodel import Session, col, select, update

from domain import task_manager
from domain.messages import ScanTaskPayload, VacuumTaskPayload
from domain.schemas import Song, TaskStatus
from worker import audio, tasks


def test_scan_skips_corrupt_file(db: Session, music_root: Path) -> None:
    valid_wav = music_root / "valid.wav"
    sample_rate = 24000
    audio_data = np.zeros(sample_rate, dtype=np.float32)
    sf.write(str(valid_wav), audio_data, sample_rate)

    bad_flac = music_root / "bad.flac"
    bad_flac.write_bytes(b"corrupt random bytes not a flac file 123456789")

    task = task_manager.create_task(db, ScanTaskPayload.NAME)
    tasks.scan_library_task(str(task.id), {"task_id": task.id, "target_path": "."})

    db.expire_all()
    refreshed_task = task_manager.get_task(db, task.id)
    assert refreshed_task is not None
    assert refreshed_task.status == TaskStatus.SUCCESS
    assert "Skipped (unreadable): 1" in (refreshed_task.message or "")

    songs = db.exec(select(Song)).all()
    assert len(songs) == 1
    assert songs[0].file_path == "valid.wav"


def test_vacuum_keeps_unreadable_file(
    db: Session,
    music_root: Path,
    make_song: Callable[..., str],
    fake_vector_store: Any,
) -> None:
    rel_path = make_song("album/track.wav")
    full_path = music_root / rel_path

    old_mtime = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=10)
    db.exec(update(Song).where(col(Song.file_path) == rel_path).values(mtime=old_mtime))
    db.commit()

    full_path.write_bytes(b"corrupt random bytes not a wav file 987654321")

    task = task_manager.create_task(db, VacuumTaskPayload.NAME)
    tasks.vacuum_library_task(str(task.id), {"task_id": task.id})

    db.expire_all()
    refreshed_task = task_manager.get_task(db, task.id)
    assert refreshed_task is not None
    assert refreshed_task.status == TaskStatus.SUCCESS
    assert "Skipped (unreadable): 1" in (refreshed_task.message or "")

    song = db.exec(select(Song).where(col(Song.file_path) == rel_path)).one_or_none()
    assert song is not None


def test_vacuum_deletes_missing_file(
    db: Session,
    music_root: Path,
    make_song: Callable[..., str],
    fake_vector_store: Any,
) -> None:
    rel_path = make_song("album/deleted_track.wav")
    full_path = music_root / rel_path

    full_path.unlink()

    task = task_manager.create_task(db, VacuumTaskPayload.NAME)
    tasks.vacuum_library_task(str(task.id), {"task_id": task.id})

    db.expire_all()
    refreshed_task = task_manager.get_task(db, task.id)
    assert refreshed_task is not None
    assert refreshed_task.status == TaskStatus.SUCCESS
    assert "Deleted: 1" in (refreshed_task.message or "")

    song = db.exec(select(Song).where(col(Song.file_path) == rel_path)).one_or_none()
    assert song is None


def test_check_audio_file_status_unreadable_on_mtime_oserror(
    music_root: Path,
    monkeypatch: Any,
) -> None:
    test_file = music_root / "test.wav"
    test_file.touch()

    def mock_getmtime(p: Any) -> float:
        raise PermissionError("Access denied")

    monkeypatch.setattr("os.path.getmtime", mock_getmtime)
    res = audio.check_audio_file_status(test_file, None)
    assert res.status == "UNREADABLE"


def test_check_audio_file_status_unreadable_on_corrupt_file(
    music_root: Path,
) -> None:
    bad_file = music_root / "bad.flac"
    bad_file.write_bytes(b"not a valid flac")
    res = audio.check_audio_file_status(bad_file, None)
    assert res.status == "UNREADABLE"


def test_extract_metadata_nonexistent_file(tmp_path: Path) -> None:
    res = audio.extract_metadata(tmp_path / "nonexistent.flac")
    assert res is None


def test_extract_metadata_corrupt_file(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.flac"
    bad.write_bytes(b"garbage")
    res = audio.extract_metadata(bad)
    assert res is None


def test_scan_task_chunk_exception_skips_files(
    db: Session,
    music_root: Path,
    monkeypatch: Any,
) -> None:
    test_file = music_root / "file.wav"
    test_file.touch()

    def mock_extract_bulk(chunks: Any) -> Any:
        raise RuntimeError("Unexpected failure in worker")

    monkeypatch.setattr(audio, "extract_metadata_bulk", mock_extract_bulk)

    task = task_manager.create_task(db, ScanTaskPayload.NAME)
    tasks.scan_library_task(str(task.id), {"task_id": task.id, "target_path": "."})

    db.expire_all()
    refreshed_task = task_manager.get_task(db, task.id)
    assert refreshed_task is not None
    assert refreshed_task.status == TaskStatus.SUCCESS
    assert "Skipped (unreadable): 1" in (refreshed_task.message or "")
    assert "Successfully scanned and processed 0 files" in (refreshed_task.message or "")


def test_vacuum_task_exception_skips_file(
    db: Session,
    music_root: Path,
    make_song: Callable[..., str],
    fake_vector_store: Any,
    monkeypatch: Any,
) -> None:
    rel_path = make_song("album/test_error.wav")

    def mock_check_status(p: Any, m: Any) -> Any:
        raise RuntimeError("Disk read error")

    monkeypatch.setattr(audio, "check_audio_file_status", mock_check_status)

    task = task_manager.create_task(db, VacuumTaskPayload.NAME)
    tasks.vacuum_library_task(str(task.id), {"task_id": task.id})

    db.expire_all()
    refreshed_task = task_manager.get_task(db, task.id)
    assert refreshed_task is not None
    assert refreshed_task.status == TaskStatus.SUCCESS
    assert "Skipped (unreadable): 1" in (refreshed_task.message or "")

    song = db.exec(select(Song).where(col(Song.file_path) == rel_path)).one_or_none()
    assert song is not None

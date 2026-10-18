import importlib
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from domain import config
from domain.enums import EmbeddingModel
from inference.download_models import (
    DOWNLOADERS,
    download_clap,
    download_model,
    get_target_path,
    main,
    resolve_models,
)


def test_default_model_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default model set is analyze models union muq_mulan minus acoustic_features."""
    monkeypatch.setattr(config, "ANALYZE_MODELS", ["muq", "muq_mulan"])
    assert set(resolve_models()) == {"muq", "muq_mulan"}

    monkeypatch.setattr(config, "ANALYZE_MODELS", ["mert", "acoustic_features"])
    assert set(resolve_models()) == {"mert", "muq_mulan"}

    monkeypatch.setattr(config, "ANALYZE_MODELS", ["acoustic_features"])
    assert set(resolve_models()) == {"muq_mulan"}


def test_models_arg_clap() -> None:
    """Passing --models clap gives {clap, muq_mulan}."""
    assert set(resolve_models("clap")) == {"clap", "muq_mulan"}
    assert set(resolve_models("clap,mert")) == {"clap", "mert", "muq_mulan"}
    assert set(resolve_models("acoustic_features")) == {"muq_mulan"}

    with pytest.raises(ValueError, match="Unknown model 'invalid_model'"):
        resolve_models("invalid_model")


def test_marked_target_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A target with a marker is skipped without calling downloader."""
    target_dir = get_target_path("muq", tmp_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "weights.bin").write_text("model weights")
    marker_file = Path(f"{target_dir}.complete")
    marker_file.touch()

    mock_dl = MagicMock()
    monkeypatch.setitem(DOWNLOADERS, "muq", mock_dl)

    result = download_model("muq", tmp_path)
    assert result == "skip"
    mock_dl.assert_not_called()


def test_non_empty_unmarked_directory_is_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-empty unmarked directory is adopted and marked without calling downloader."""
    target_dir = get_target_path("muq", tmp_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "weights.bin").write_text("model weights from old script")
    marker_file = Path(f"{target_dir}.complete")
    assert not marker_file.exists()

    mock_dl = MagicMock()
    monkeypatch.setitem(DOWNLOADERS, "muq", mock_dl)

    result = download_model("muq", tmp_path)
    assert result == "adopt existing"
    mock_dl.assert_not_called()
    assert marker_file.exists()


def test_empty_unmarked_directory_is_redownloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty unmarked directory is deleted and re-downloaded."""
    target_dir = get_target_path("muq", tmp_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    marker_file = Path(f"{target_dir}.complete")
    assert not marker_file.exists()

    def fake_download(target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / "new_weights.bin").write_text("fresh download")

    mock_dl = MagicMock(side_effect=fake_download)
    monkeypatch.setitem(DOWNLOADERS, "muq", mock_dl)

    result = download_model("muq", tmp_path)
    assert result == "downloaded"
    mock_dl.assert_called_once_with(target_dir)
    assert marker_file.exists()
    assert (target_dir / "new_weights.bin").exists()


def test_leftover_temporary_file_is_cleaned_and_redownloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leftover temporary file without marker is removed and re-downloaded."""
    target_file = get_target_path("clap", tmp_path)
    target_file.write_text("leftover partial file")
    tmp_file = Path(f"{target_file}.tmp")
    tmp_file.write_text("leftover tmp chunk")
    marker_file = Path(f"{target_file}.complete")

    def fake_download(target: Path) -> None:
        target.write_text("complete checkpoint")

    mock_dl = MagicMock(side_effect=fake_download)
    monkeypatch.setitem(DOWNLOADERS, "clap", mock_dl)

    result = download_model("clap", tmp_path)
    assert result == "downloaded"
    mock_dl.assert_called_once_with(target_file)
    assert marker_file.exists()
    assert not tmp_file.exists()
    assert target_file.read_text() == "complete checkpoint"


def test_failing_downloader_leaves_no_marker_and_main_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing downloader leaves no marker and causes main() to return non-zero."""
    target_dir = get_target_path("muq", tmp_path)
    marker_file = Path(f"{target_dir}.complete")

    def failing_download(target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / "corrupted.part").write_text("incomplete")
        raise RuntimeError("Network interrupted")

    monkeypatch.setitem(DOWNLOADERS, "muq", failing_download)
    monkeypatch.setitem(DOWNLOADERS, "muq_mulan", MagicMock())

    ret = main(["--models", "muq", "--dir", str(tmp_path)])
    assert ret != 0
    assert not marker_file.exists()


def test_config_analyze_models_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """config.ANALYZE_MODELS validates choices against EmbeddingModel values."""
    with pytest.raises(ValueError, match="Unknown model 'invalid_model' in LAINBOW_ANALYZE_MODELS"):
        config._parse_analyze_models("invalid_model")

    with pytest.raises(ValueError, match="Unknown model 'unknown' in LAINBOW_ANALYZE_MODELS"):
        config._parse_analyze_models("muq,unknown")

    # Stripping and empty items dropped
    assert config._parse_analyze_models("  muq , muq_mulan , ") == [
        EmbeddingModel.MUQ.value,
        EmbeddingModel.MUQ_MULAN.value,
    ]

    # Reloading config with invalid env variable raises ValueError
    monkeypatch.setenv("LAINBOW_ANALYZE_MODELS", "bad_model")
    with pytest.raises(ValueError, match="Unknown model 'bad_model' in LAINBOW_ANALYZE_MODELS"):
        importlib.reload(config)


def test_main_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() with mock downloaders successfully runs and returns 0."""

    def fake_muq(target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)

    def fake_mulan(target: Path) -> None:
        target.mkdir(parents=True, exist_ok=True)

    def fake_clap(target: Path) -> None:
        target.write_text("clap")

    monkeypatch.setitem(DOWNLOADERS, "muq", MagicMock(side_effect=fake_muq))
    monkeypatch.setitem(DOWNLOADERS, "muq_mulan", MagicMock(side_effect=fake_mulan))
    monkeypatch.setitem(DOWNLOADERS, "clap", MagicMock(side_effect=fake_clap))

    ret = main(["--models", "clap", "--dir", str(tmp_path)])
    assert ret == 0

    assert (tmp_path / "music_audioset_epoch_15_esc_90.14.pt.complete").exists()
    assert (tmp_path / "MuQ-MuLan-large.complete").exists()


def test_main_invalid_models_arg(tmp_path: Path) -> None:
    """main() with invalid model arg returns non-zero."""
    ret = main(["--models", "unknown_xyz", "--dir", str(tmp_path)])
    assert ret != 0


class DummyResponse:
    def __init__(self, chunks: list[bytes], status_code: int = 200) -> None:
        self.chunks = chunks
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=MagicMock(), response=MagicMock())

    def iter_bytes(self, chunk_size: int = 8192) -> list[bytes]:
        return self.chunks

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, *args: object) -> None:
        pass


def test_download_clap_streaming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """download_clap streams via a temporary file and renames on success."""
    target = tmp_path / "model.pt"

    def mock_stream(method: str, url: str, **kwargs: object) -> DummyResponse:
        return DummyResponse([b"chunk1", b"chunk2"])

    monkeypatch.setattr(httpx, "stream", mock_stream)
    download_clap(target)

    assert target.exists()
    assert target.read_bytes() == b"chunk1chunk2"
    assert not Path(f"{target}.tmp").exists()


def test_download_clap_failure_cleans_up_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """download_clap cleans up .tmp file if an error occurs."""
    target = tmp_path / "model.pt"

    def mock_stream(method: str, url: str, **kwargs: object) -> DummyResponse:
        return DummyResponse([], status_code=500)

    monkeypatch.setattr(httpx, "stream", mock_stream)
    with pytest.raises(httpx.HTTPStatusError):
        download_clap(target)

    assert not target.exists()
    assert not Path(f"{target}.tmp").exists()

"""WI-M3-03 (Critical Module, invariant I2): `./lainbow setup` must never lose or change a user's settings.

The script is driven through its test hooks: LAINBOW_ENV_FILE points it at a temporary .env,
LAINBOW_SKIP_PREREQS=1 skips the Docker checks, and --no-download skips fetching models.
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAINBOW = REPO_ROOT / "lainbow"


def _setup(env_file: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "LAINBOW_ENV_FILE": str(env_file),
        "LAINBOW_SKIP_PREREQS": "1",
    }
    return subprocess.run(
        [str(LAINBOW), "setup", "--yes", "--no-download", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _values(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in env_file.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, value = stripped.split("=", 1)
            values[key] = value
    return values


@pytest.fixture
def music_dir(tmp_path: Path) -> Path:
    path = tmp_path / "music"
    path.mkdir()
    return path


def test_fresh_setup_writes_complete_env(tmp_path: Path, music_dir: Path) -> None:
    env_file = tmp_path / ".env"

    result = _setup(env_file, "--music-dir", str(music_dir), "--accel", "cpu")

    assert result.returncode == 0, result.stderr
    values = _values(env_file)
    assert values["COMPOSE_PROFILES"] == "db,app,inference-cpu"
    assert values["MUSIC_NAS_ROOT_DIR"] == str(music_dir)
    assert values["LAINBOW_ANALYZE_MODELS"] == "muq,muq_mulan"
    assert len(values["POSTGRES_PASSWORD"]) >= 16
    assert len(values["RABBITMQ_DEFAULT_PASS"]) >= 16
    assert values["POSTGRES_PASSWORD"] != values["RABBITMQ_DEFAULT_PASS"]
    # CPU only changes performance settings, never what is computed (decision D-F)
    assert values["TASK_CHUNK_SIZE"] == "2"
    assert values["INFERENCE_REQUEST_TIMEOUT"] == "600"
    assert values["TASK_STALE_AFTER_SEC"] == "7200"
    assert "MUQ_FILE_DURATION_SEC" not in values
    # The file holds passwords
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600


def test_gpu_setup_uses_the_same_models_without_cpu_tuning(tmp_path: Path, music_dir: Path) -> None:
    env_file = tmp_path / ".env"

    result = _setup(env_file, "--music-dir", str(music_dir), "--accel", "cuda")

    assert result.returncode == 0, result.stderr
    values = _values(env_file)
    assert values["COMPOSE_PROFILES"] == "db,app,inference-cuda"
    assert values["LAINBOW_ANALYZE_MODELS"] == "muq,muq_mulan"
    assert "TASK_CHUNK_SIZE" not in values


@pytest.mark.skipif(not Path("/dev/kfd").exists(), reason="needs an AMD GPU device node")
def test_rocm_setup_records_the_device_gid(tmp_path: Path, music_dir: Path) -> None:
    env_file = tmp_path / ".env"

    result = _setup(env_file, "--music-dir", str(music_dir), "--accel", "rocm")

    assert result.returncode == 0, result.stderr
    values = _values(env_file)
    assert values["COMPOSE_PROFILES"] == "db,app,inference-rocm"
    assert values["LAINBOW_GPU_GID"] == str(Path("/dev/kfd").stat().st_gid)


def test_second_run_changes_nothing(tmp_path: Path, music_dir: Path) -> None:
    env_file = tmp_path / ".env"
    assert _setup(env_file, "--music-dir", str(music_dir), "--accel", "cpu").returncode == 0
    first = env_file.read_text()

    result = _setup(env_file, "--music-dir", str(music_dir), "--accel", "cpu")

    assert result.returncode == 0, result.stderr
    assert env_file.read_text() == first


def test_existing_values_win_over_new_arguments(tmp_path: Path, music_dir: Path) -> None:
    env_file = tmp_path / ".env"
    assert _setup(env_file, "--music-dir", str(music_dir), "--accel", "cpu").returncode == 0
    before = _values(env_file)
    other_music = tmp_path / "other"
    other_music.mkdir()

    result = _setup(env_file, "--music-dir", str(other_music), "--accel", "cuda")

    assert result.returncode == 0, result.stderr
    after = _values(env_file)
    assert after == before
    # It must say so rather than silently ignoring the arguments
    assert "COMPOSE_PROFILES" in result.stdout + result.stderr


def test_legacy_env_keeps_every_line_and_gets_the_old_credentials(
    tmp_path: Path, music_dir: Path
) -> None:
    """An .env made from the pre-M3 template: no passwords, old variable names, comments."""
    env_file = tmp_path / ".env"
    legacy = (
        "#### Music library folder #####\n"
        f"MUSIC_NAS_ROOT_DIR={music_dir}\n"
        "#WEB_API_PORT=8000\n"
        "DATABASE_SERVER_ENDPOINT=host.docker.internal\n"
        "POSTGRES_PORT=5433\n"
        "INFERENCE_MODEL_MANAGER_MODE=single\n"
        "TASK_CHUNK_SIZE=10\n"
    )
    env_file.write_text(legacy)

    result = _setup(env_file, "--accel", "cpu")

    assert result.returncode == 0, result.stderr
    text = env_file.read_text()
    # Every original line survives verbatim and in order, at the top
    assert text.startswith(legacy)
    values = _values(env_file)
    # The databases of an existing install were created with the old hard-coded credentials
    assert values["POSTGRES_PASSWORD"] == "password"
    assert values["RABBITMQ_DEFAULT_PASS"] == "password"
    # The user's own tuning is kept even though CPU setup would choose 2
    assert values["TASK_CHUNK_SIZE"] == "10"
    assert values["COMPOSE_PROFILES"] == "db,app,inference-cpu"


def test_missing_music_dir_fails_without_writing(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"

    result = _setup(env_file, "--music-dir", str(tmp_path / "nope"), "--accel", "cpu")

    assert result.returncode != 0
    assert not env_file.exists()


def test_backup_is_made_before_changing_an_existing_file(tmp_path: Path, music_dir: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"MUSIC_NAS_ROOT_DIR={music_dir}\n")

    assert _setup(env_file, "--accel", "cpu").returncode == 0

    backups = list(tmp_path.glob(".env.bak*"))
    assert len(backups) == 1
    assert backups[0].read_text() == f"MUSIC_NAS_ROOT_DIR={music_dir}\n"

"""Tests for the ./lainbow helper, run against a fake ``docker`` on PATH."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from lainbow.server.engines import parse_token

ROOT = Path(__file__).resolve().parents[2]
FAKE_TOKEN = "lbe_" + "ab" * 16 + "_" + "x" * 43

FAKE_DOCKER = f"""#!/bin/sh
echo "$@" >> "$FAKE_DOCKER_LOG"
case "$*" in
"compose version") echo "Docker Compose version v2.0.0" ;;
*"lainbow-admin add-engine"*) echo "{FAKE_TOKEN}" ;;
esac
exit 0
"""


@pytest.fixture
def work(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy(ROOT / "lainbow", project / "lainbow")
    shutil.copy(ROOT / "compose.yaml", project / "compose.yaml")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "music").mkdir()
    return project


def run(work: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PATH": f"{work.parent / 'bin'}:{os.environ['PATH']}",
        "FAKE_DOCKER_LOG": str(work.parent / "docker.log"),
    }
    return subprocess.run(
        [str(work / "lainbow"), *args],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
        check=False,
    )


def read_env(work: Path) -> dict[str, str]:
    lines = (work / ".env").read_text().splitlines()
    return {line.partition("=")[0]: line.partition("=")[2] for line in lines if line}


def test_init_writes_every_key(work: Path) -> None:
    music = work.parent / "music"
    result = run(work, "init", "--yes", "--music-dir", str(music), "--accel", "cpu")
    assert result.returncode == 0, result.stderr
    env = read_env(work)
    assert env["MUSIC_DIR"] == str(music)
    assert env["DATA_DIR"] == "./data"
    assert env["PORT"] == "8000"
    assert env["COMPOSE_PROFILES"] == "cpu"
    assert re.fullmatch(r"[A-Za-z0-9_-]{32}", env["POSTGRES_PASSWORD"])
    assert re.fullmatch(r"lbe_[0-9a-f]{32}_[A-Za-z0-9_-]{43}", env["ENGINE_TOKEN"])
    parse_token(env["ENGINE_TOKEN"])
    assert "docker compose up -d" in result.stdout


def test_init_twice_keeps_everything(work: Path) -> None:
    music = str(work.parent / "music")
    assert run(work, "init", "--yes", "--music-dir", music, "--accel", "cpu").returncode == 0
    before = (work / ".env").read_bytes()
    result = run(work, "init", "--yes")
    assert result.returncode == 0, result.stderr
    assert (work / ".env").read_bytes() == before


def test_init_adds_only_missing_key(work: Path) -> None:
    music = work.parent / "music"
    original = (
        f"# my settings\nMUSIC_DIR={music}\nDATA_DIR=/srv/lainbow\nPORT=9000\n"
        "COMPOSE_PROFILES=cuda\nPOSTGRES_PASSWORD=secret"
    )
    (work / ".env").write_text(original)
    result = run(work, "init", "--yes")
    assert result.returncode == 0, result.stderr
    content = (work / ".env").read_text()
    assert content.startswith(original + "\n")
    added = content[len(original) + 1 :]
    assert re.fullmatch(r"ENGINE_TOKEN=lbe_[0-9a-f]{32}_[A-Za-z0-9_-]{43}\n", added)
    assert "ENGINE_TOKEN" in result.stdout


def test_init_missing_music_dir_fails(work: Path) -> None:
    result = run(work, "init", "--yes", "--music-dir", str(work / "nope"), "--accel", "cpu")
    assert result.returncode != 0
    assert "does not exist" in result.stderr
    assert not (work / ".env").exists()


def test_add_engine_cuda(work: Path) -> None:
    (work / ".env").write_text("PORT=9000\nLAINBOW_VERSION=2.1\n")
    result = run(work, "add-engine", "desktop", "--accel", "cuda", "--host", "10.0.0.5")
    assert result.returncode == 0, result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("docker run")]
    assert len(commands) == 1
    cmd = commands[0]
    assert "--gpus all" in cmd
    assert "LAINBOW_SERVER_URL=http://10.0.0.5:9000" in cmd
    assert f"LAINBOW_TOKEN={FAKE_TOKEN}" in cmd
    assert cmd.endswith("ghcr.io/prokosna/lainbow-engine:2.1-cuda")
    assert "docker rm -f lainbow-engine" in result.stdout
    assert "add-engine desktop" in (work.parent / "docker.log").read_text()


def test_add_engine_cpu_has_no_gpu(work: Path) -> None:
    result = run(work, "add-engine", "box", "--accel", "cpu", "--host", "h")
    assert result.returncode == 0, result.stderr
    assert "--gpus" not in result.stdout
    assert "LAINBOW_SERVER_URL=http://h:8000" in result.stdout
    assert result.stdout.count("lainbow-engine:2-cpu") == 1


def test_add_engine_rocm_devices(work: Path) -> None:
    result = run(work, "add-engine", "amd", "--accel", "rocm", "--host", "h")
    assert "--device /dev/kfd --device /dev/dri" in result.stdout


def test_status_and_update_call_compose(work: Path) -> None:
    assert run(work, "status").returncode == 0
    assert run(work, "config").returncode == 0
    assert run(work, "update").returncode == 0
    log = (work.parent / "docker.log").read_text()
    assert "lainbow-admin status" in log
    assert "lainbow-admin config" in log
    assert "compose pull" in log
    assert "compose up -d" in log
    assert " -p " not in log


def test_help_and_unknown(work: Path) -> None:
    assert "add-engine" in run(work).stdout
    assert run(work, "bogus").returncode != 0

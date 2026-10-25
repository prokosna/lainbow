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
"info --format {{{{json .Runtimes}}}}") echo '{{"nvidia":{{}},"runc":{{}}}}' ;;
"ps -a"*) [ -n "$FAKE_EXISTING" ] && echo "$FAKE_EXISTING" ;;
"pull "*) [ -n "$FAKE_PULL_FAIL" ] && exit 1 ;;
esac
exit 0
"""


FAKE_CURL = """#!/bin/sh
echo "$@" >> "$FAKE_CURL_LOG"
case "$*" in
*/health*)
    [ "$FAKE_CURL_FAIL" = health ] && exit 22
    echo "{\\"status\\":\\"ok\\",\\"version\\":\\"$FAKE_SERVER_VERSION\\"}"
    ;;
*/engine/v1/plan*)
    [ "$FAKE_CURL_FAIL" = plan ] && exit 22
    echo "{}"
    ;;
esac
exit 0
"""

FAKE_NVIDIA_SMI = "#!/bin/sh\nexit 0\n"

JOIN_URL = "http://10.0.0.5:8000"
JOIN_TOKEN = "lbe_tok"


def install_fake(work: Path, name: str, body: str) -> None:
    path = work.parent / "bin" / name
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def docker_calls(work: Path) -> list[str]:
    log = work.parent / "docker.log"
    return log.read_text().splitlines() if log.exists() else []


@pytest.fixture
def work(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy(ROOT / "lainbow", project / "lainbow")
    shutil.copy(ROOT / "compose.yaml", project / "compose.yaml")
    (project / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "9.9.9"\n')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    install_fake(project, "curl", FAKE_CURL)
    (tmp_path / "music").mkdir()
    return project


def run(
    work: Path, *args: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PATH": f"{work.parent / 'bin'}:{os.environ['PATH']}",
        "FAKE_DOCKER_LOG": str(work.parent / "docker.log"),
        "FAKE_CURL_LOG": str(work.parent / "curl.log"),
        "FAKE_SERVER_VERSION": "9.9.9",
        "FAKE_CURL_FAIL": "",
        "FAKE_EXISTING": "",
        "FAKE_PULL_FAIL": "",
        "LAINBOW_KFD_DEVICE": str(work.parent / "no-kfd"),
        "HSA_OVERRIDE_GFX_VERSION": "",
        **(extra_env or {}),
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


def test_add_engine_prints_join_line(work: Path) -> None:
    (work / ".env").write_text("PORT=9000\n")
    result = run(work, "add-engine", "desktop", "--host", "10.0.0.5")
    assert result.returncode == 0, result.stderr
    assert f"./lainbow join http://10.0.0.5:9000 {FAKE_TOKEN}" in result.stdout
    assert "WSL" in result.stdout
    assert "add-engine desktop" in (work.parent / "docker.log").read_text()
    assert run(work, "add-engine", "x", "--accel", "cpu").returncode != 0


def test_join_cuda(work: Path) -> None:
    install_fake(work, "nvidia-smi", FAKE_NVIDIA_SMI)
    result = run(work, "join", JOIN_URL, JOIN_TOKEN)
    assert result.returncode == 0, result.stderr
    assert "accelerator: cuda (nvidia-smi found)" in result.stdout
    runs = [c for c in docker_calls(work) if c.startswith("run ")]
    assert len(runs) == 1
    assert "--gpus all" in runs[0]
    assert f"LAINBOW_SERVER_URL={JOIN_URL}" in runs[0]
    assert f"LAINBOW_TOKEN={JOIN_TOKEN}" in runs[0]
    assert runs[0].endswith("ghcr.io/prokosna/lainbow-engine:9.9.9-cuda")
    assert "--name lainbow-engine" in runs[0]
    assert "pull ghcr.io/prokosna/lainbow-engine:9.9.9-cuda" in docker_calls(work)
    assert "docker logs -f lainbow-engine" in result.stdout


def test_join_cpu_has_no_gpu(work: Path) -> None:
    install_fake(work, "nvidia-smi", FAKE_NVIDIA_SMI)
    result = run(work, "join", JOIN_URL, JOIN_TOKEN, "--accel", "cpu")
    assert result.returncode == 0, result.stderr
    runs = [c for c in docker_calls(work) if c.startswith("run ")]
    assert "--gpus" not in runs[0]
    assert runs[0].endswith("lainbow-engine:9.9.9-cpu")


def test_join_rocm(work: Path) -> None:
    kfd = work.parent / "kfd"
    kfd.write_text("")
    env = {"LAINBOW_KFD_DEVICE": str(kfd)}
    result = run(work, "join", JOIN_URL, JOIN_TOKEN, extra_env=env)
    assert result.returncode == 0, result.stderr
    assert "accelerator: rocm" in result.stdout
    runs = [c for c in docker_calls(work) if c.startswith("run ")]
    assert "--device /dev/kfd --device /dev/dri" in runs[0]
    assert f"--group-add {kfd.stat().st_gid}" in runs[0]
    assert "MIOPEN_FIND_MODE=FAST" in runs[0]
    assert "HSA_OVERRIDE_GFX_VERSION" not in runs[0]

    (work.parent / "docker.log").unlink()
    env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    run(work, "join", JOIN_URL, JOIN_TOKEN, "--yes", extra_env=env)
    runs = [c for c in docker_calls(work) if c.startswith("run ")]
    assert "-e HSA_OVERRIDE_GFX_VERSION=10.3.0" in runs[0]


def test_join_version_mismatch(work: Path) -> None:
    result = run(
        work,
        "join",
        JOIN_URL,
        JOIN_TOKEN,
        "--accel",
        "cpu",
        extra_env={"FAKE_SERVER_VERSION": "2.1.0"},
    )
    assert result.returncode != 0
    assert (
        "server is 2.1.0 but this checkout is 9.9.9; "
        "check out the same release (git checkout v2.1.0)" in result.stderr
    )
    assert docker_calls(work) == []


def test_join_bad_token(work: Path) -> None:
    result = run(
        work, "join", JOIN_URL, JOIN_TOKEN, "--accel", "cpu", extra_env={"FAKE_CURL_FAIL": "plan"}
    )
    assert result.returncode != 0
    assert "the server rejected the token; run ./lainbow add-engine NAME on the server again" in (
        result.stderr
    )
    assert not any(c.startswith("run ") for c in docker_calls(work))


def test_join_unreachable(work: Path) -> None:
    result = run(
        work, "join", JOIN_URL, JOIN_TOKEN, "--accel", "cpu", extra_env={"FAKE_CURL_FAIL": "health"}
    )
    assert result.returncode != 0
    assert f"cannot reach the server at {JOIN_URL}; is it running and is the port open?" in (
        result.stderr
    )


def test_join_builds_when_pull_fails(work: Path) -> None:
    result = run(
        work, "join", JOIN_URL, JOIN_TOKEN, "--accel", "cuda", extra_env={"FAKE_PULL_FAIL": "1"}
    )
    assert result.returncode == 0, result.stderr
    calls = docker_calls(work)
    build = [c for c in calls if c.startswith("build ")]
    assert len(build) == 1
    assert "--build-arg ACCEL=cuda" in build[0]
    assert "-t ghcr.io/prokosna/lainbow-engine:9.9.9-cuda" in build[0]
    assert "-f docker/engine.Dockerfile" in build[0]
    assert any(c.startswith("run ") for c in calls)


def test_join_build_flag_skips_pull(work: Path) -> None:
    result = run(work, "join", JOIN_URL, JOIN_TOKEN, "--accel", "cpu", "--build")
    assert result.returncode == 0, result.stderr
    calls = docker_calls(work)
    assert not any(c.startswith("pull ") for c in calls)
    assert any(c.startswith("build ") for c in calls)


def test_join_print_runs_nothing(work: Path) -> None:
    result = run(work, "join", JOIN_URL, JOIN_TOKEN, "--accel", "cpu", "--print")
    assert result.returncode == 0, result.stderr
    assert "docker run -d --name lainbow-engine" in result.stdout
    assert f"LAINBOW_TOKEN={JOIN_TOKEN}" in result.stdout
    calls = docker_calls(work)
    assert not any(c.split(" ")[0] in ("run", "pull", "build") for c in calls)


def test_join_replaces_existing_container(work: Path) -> None:
    result = run(
        work,
        "join",
        JOIN_URL,
        JOIN_TOKEN,
        "--accel",
        "cpu",
        "--yes",
        extra_env={"FAKE_EXISTING": "lainbow-engine"},
    )
    assert result.returncode == 0, result.stderr
    calls = docker_calls(work)
    rm = calls.index("rm -f lainbow-engine")
    assert calls[rm + 1].startswith("run ")

    refused = run(
        work,
        "join",
        JOIN_URL,
        JOIN_TOKEN,
        "--accel",
        "cpu",
        extra_env={"FAKE_EXISTING": "lainbow-engine"},
    )
    assert refused.returncode != 0


def test_leave(work: Path) -> None:
    result = run(work, "leave")
    assert result.returncode == 0, result.stderr
    assert "rm -f lainbow-engine" in docker_calls(work)
    assert "docker volume rm lainbow-models" in result.stdout


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
    out = run(work).stdout
    assert "add-engine" in out
    assert "join" in out
    assert "leave" in out
    assert run(work, "bogus").returncode != 0

"""Tests for the ./lainbow helper, run against a fake ``docker`` on PATH."""

from __future__ import annotations

import os
import re
import shutil
import socketserver
import stat
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
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
"image inspect "*) [ -n "$FAKE_IMAGE_EXISTS" ] || exit 1 ;;
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

FAKE_GIT = """#!/bin/sh
echo "$@" >> "$FAKE_GIT_LOG"
case "$1" in
describe) echo v2.0.0 ;;
tag) printf 'v2.1.0\\nv2.0.0\\n' ;;
esac
exit 0
"""

# Reports a listener only for the ports named in FAKE_PORTS_BUSY (real ss prints a header).
FAKE_SS = """#!/bin/sh
echo "State Recv-Q Send-Q Local Address:Port Peer Address:Port"
for p in $FAKE_PORTS_BUSY; do
    case "$*" in
    *":$p") echo "LISTEN 0 4096 0.0.0.0:$p 0.0.0.0:*" ;;
    esac
done
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
    install_fake(project, "ss", FAKE_SS)
    (tmp_path / "music").mkdir()
    return project


def run(
    work: Path, *args: str, extra_env: dict[str, str] | None = None, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PATH": f"{work.parent / 'bin'}:{os.environ['PATH']}",
        "FAKE_DOCKER_LOG": str(work.parent / "docker.log"),
        "FAKE_GIT_LOG": str(work.parent / "git.log"),
        "FAKE_CURL_LOG": str(work.parent / "curl.log"),
        "FAKE_SERVER_VERSION": "9.9.9",
        "FAKE_CURL_FAIL": "",
        "FAKE_EXISTING": "",
        "FAKE_IMAGE_EXISTS": "",
        "LAINBOW_KFD_DEVICE": str(work.parent / "no-kfd"),
        "HSA_OVERRIDE_GFX_VERSION": "",
        "FAKE_PORTS_BUSY": "",
        **(extra_env or {}),
    }
    return subprocess.run(
        [str(work / "lainbow"), *args],
        capture_output=True,
        text=True,
        env=env,
        input=stdin,
        stdin=None if stdin is not None else subprocess.DEVNULL,
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


def make_songs(root: Path, *uris: str) -> None:
    for uri in uris:
        path = root / uri
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")


@contextmanager
def fake_mpd(uris: list[str], *, find_works: bool = True) -> Iterator[int]:
    """A tiny MPD that greets, answers find/listall and closes; yields its port."""

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            self.wfile.write(b"OK MPD 0.23.5\n")
            for raw in self.rfile:
                cmd = raw.decode().strip()
                if cmd == "close":
                    return
                if cmd.startswith("find") and find_works:
                    body = "".join(f"file: {u}\n" for u in uris)
                    self.wfile.write(body.encode() + b"OK\n")
                elif cmd.startswith("find"):
                    self.wfile.write(b"ACK [5@0] {find} unknown filter\n")
                elif cmd == "listall":
                    body = "".join(f"directory: d\nfile: {u}\n" for u in uris)
                    self.wfile.write(body.encode() + b"OK\n")

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler) as server:
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_address[1]
        finally:
            server.shutdown()


def test_init_defaults_at_every_prompt(work: Path) -> None:
    music = work.parent / "music"
    result = run(work, "init", "--accel", "cpu", stdin=f"{music}\n\n\n\n")
    assert result.returncode == 0, result.stderr
    env = read_env(work)
    assert env["MUSIC_DIR"] == str(music)
    assert env["DATA_DIR"] == "./data"
    assert env["PORT"] == "8000"
    assert "Data folder (database and model weights) [./data]: " in result.stdout
    assert "API port (Sola MPD connects here) [8000]: " in result.stdout
    assert "MPD address (host:port" in result.stdout
    assert "MPD check:    not checked" in result.stdout
    assert (work / "data").is_dir()


def test_init_custom_data_folder_is_created(work: Path) -> None:
    music = work.parent / "music"
    target = work.parent / "disk" / "lainbow"
    result = run(work, "init", "--accel", "cpu", stdin=f"{music}\n{target}\ny\n9100\n\n")
    assert result.returncode == 0, result.stderr
    assert f"Create {target}? [Y/n]" in result.stdout
    assert target.is_dir()
    env = read_env(work)
    assert env["DATA_DIR"] == str(target)
    assert env["PORT"] == "9100"


def test_init_data_folder_cannot_be_created(work: Path) -> None:
    blocker = work.parent / "file"
    blocker.write_text("")
    result = run(
        work, "init", "--yes", "--music-dir", str(work.parent / "music"), "--accel", "cpu",
        "--data-dir", str(blocker / "sub"),
    )  # fmt: skip
    assert result.returncode != 0
    assert "sudo mkdir" in result.stderr
    assert "sudo chown" in result.stderr
    assert not (work / ".env").exists()


def test_init_port_in_use_asks_again(work: Path) -> None:
    music = work.parent / "music"
    result = run(
        work,
        "init",
        "--accel",
        "cpu",
        extra_env={"FAKE_PORTS_BUSY": "8000"},
        stdin=f"{music}\n\n\n\n9200\n\n",
    )
    assert result.returncode == 0, result.stderr
    assert "Port 8000 is already in use" in result.stdout
    assert read_env(work)["PORT"] == "9200"


def test_init_yes_port_in_use_fails(work: Path) -> None:
    result = run(
        work, "init", "--yes", "--music-dir", str(work.parent / "music"), "--accel", "cpu",
        extra_env={"FAKE_PORTS_BUSY": "8000"},
    )  # fmt: skip
    assert result.returncode != 0
    assert "port 8000 is already in use" in result.stderr
    assert not (work / ".env").exists()


def test_init_mpd_paths_match(work: Path) -> None:
    music = work.parent / "music"
    uris = ["Artist/a.mp3", "Artist/b.mp3", "Other/c.flac"]
    make_songs(music, *uris)
    with fake_mpd(uris) as port:
        result = run(
            work, "init", "--yes", "--music-dir", str(music), "--accel", "cpu",
            "--mpd", f"127.0.0.1:{port}",
        )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert "paths match MPD (3/3 sample songs found)" in result.stdout


def test_init_mpd_listall_fallback(work: Path) -> None:
    music = work.parent / "music"
    uris = ["Artist/a.mp3", "Other/c.flac"]
    make_songs(music, *uris)
    with fake_mpd(uris, find_works=False) as port:
        result = run(
            work, "init", "--yes", "--music-dir", str(music), "--accel", "cpu",
            "--mpd", f"127.0.0.1:{port}",
        )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert "paths match MPD (2/2 sample songs found)" in result.stdout


def test_init_mpd_folder_too_high(work: Path) -> None:
    music = work.parent / "music"
    uris = ["Artist/a.mp3", "Artist/b.mp3"]
    make_songs(music / "library", *uris)
    with fake_mpd(uris) as port:
        result = run(work, "init", "--accel", "cpu", stdin=f"{music}\n\n\n\n127.0.0.1:{port}\ny\n")
    assert result.returncode == 0, result.stderr
    assert "Use the suggested folder? [Y/n]" in result.stdout
    assert read_env(work)["MUSIC_DIR"] == str(music / "library")


def test_init_mpd_folder_too_high_with_yes_fails(work: Path) -> None:
    music = work.parent / "music"
    uris = ["Artist/a.mp3"]
    make_songs(music / "library", *uris)
    with fake_mpd(uris) as port:
        result = run(
            work, "init", "--yes", "--music-dir", str(music), "--accel", "cpu",
            "--mpd", f"127.0.0.1:{port}",
        )  # fmt: skip
    assert result.returncode != 0
    assert str(music / "library") in result.stderr
    assert not (work / ".env").exists()


def test_init_mpd_folder_too_deep(work: Path) -> None:
    library = work.parent / "music" / "library"
    uris = ["Artist/a.mp3", "Artist/b.mp3"]
    make_songs(library, *uris)
    deep = library / "Artist"
    with fake_mpd(uris) as port:
        result = run(work, "init", "--accel", "cpu", stdin=f"{deep}\n\n\n\n127.0.0.1:{port}\ny\n")
    assert result.returncode == 0, result.stderr
    assert f"{library}" in result.stderr
    assert read_env(work)["MUSIC_DIR"] == str(library)


def test_init_mpd_unrelated_folder_shows_examples(work: Path) -> None:
    music = work.parent / "music"
    with fake_mpd(["Artist/a.mp3"]) as port:
        result = run(work, "init", "--accel", "cpu", stdin=f"{music}\n\n\n\n127.0.0.1:{port}\n")
    assert result.returncode == 0, result.stderr
    assert "Artist/a.mp3" in result.stderr
    assert "music_directory" in result.stderr
    assert read_env(work)["MUSIC_DIR"] == str(music)


def test_init_mpd_unreachable_warns(work: Path) -> None:
    with socketserver.TCPServer(("127.0.0.1", 0), socketserver.BaseRequestHandler) as free:
        port = free.server_address[1]
    result = run(
        work, "init", "--yes", "--music-dir", str(work.parent / "music"), "--accel", "cpu",
        "--mpd", f"127.0.0.1:{port}",
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert "warning: cannot read from MPD" in result.stderr
    assert (work / ".env").exists()


def test_init_yes_without_mpd_flag_does_not_connect(work: Path) -> None:
    result = run(work, "init", "--yes", "--music-dir", str(work.parent / "music"), "--accel", "cpu")
    assert result.returncode == 0, result.stderr
    assert "MPD address" not in result.stdout
    assert "MPD check:    not checked" in result.stdout


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
    assert "LAINBOW_TEXT_PORT=0" in runs[0]
    assert runs[0].endswith("lainbow-engine:9.9.9-cuda")
    assert "--name lainbow-engine" in runs[0]
    assert not any(c.startswith("pull ") for c in docker_calls(work))
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


def test_join_builds_when_image_missing(work: Path) -> None:
    result = run(work, "join", JOIN_URL, JOIN_TOKEN, "--accel", "cuda")
    assert result.returncode == 0, result.stderr
    calls = docker_calls(work)
    build = [c for c in calls if c.startswith("build ")]
    assert len(build) == 1
    assert "--build-arg ACCEL=cuda" in build[0]
    assert "-t lainbow-engine:9.9.9-cuda" in build[0]
    assert "-f docker/engine.Dockerfile" in build[0]
    assert any(c.startswith("run ") for c in calls)
    assert not any(c.startswith("pull ") for c in calls)


def test_join_reuses_existing_image(work: Path) -> None:
    result = run(
        work,
        "join",
        JOIN_URL,
        JOIN_TOKEN,
        "--accel",
        "cpu",
        extra_env={"FAKE_IMAGE_EXISTS": "1"},
    )
    assert result.returncode == 0, result.stderr
    calls = docker_calls(work)
    assert not any(c.startswith(("build ", "pull ")) for c in calls)
    assert any(c.startswith("run ") for c in calls)


def test_join_build_flag_forces_build(work: Path) -> None:
    result = run(
        work,
        "join",
        JOIN_URL,
        JOIN_TOKEN,
        "--accel",
        "cpu",
        "--build",
        extra_env={"FAKE_IMAGE_EXISTS": "1"},
    )
    assert result.returncode == 0, result.stderr
    calls = docker_calls(work)
    assert not any(c.startswith("pull ") for c in calls)
    assert any(c.startswith("build ") for c in calls)


def test_join_print_shows_build_line_when_missing(work: Path) -> None:
    result = run(work, "join", JOIN_URL, JOIN_TOKEN, "--accel", "cpu", "--print")
    assert "docker build -f docker/engine.Dockerfile --build-arg ACCEL=cpu" in result.stdout
    with_image = run(
        work,
        "join",
        JOIN_URL,
        JOIN_TOKEN,
        "--accel",
        "cpu",
        "--print",
        extra_env={"FAKE_IMAGE_EXISTS": "1"},
    )
    assert "docker build" not in with_image.stdout


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
    result = run(work, "leave", extra_env={"FAKE_EXISTING": "lainbow-engine"})
    assert result.returncode == 0, result.stderr
    calls = docker_calls(work)
    assert calls.index("stop -t 120 lainbow-engine") < calls.index("rm lainbow-engine")
    assert not any(c.startswith("rm -f") for c in calls)
    assert "stopping lainbow-engine" in result.stdout
    assert "docker volume rm lainbow-models" in result.stdout


def test_leave_without_container(work: Path) -> None:
    result = run(work, "leave")
    assert result.returncode == 0, result.stderr
    assert "nothing to stop" in result.stdout
    assert not any(c.startswith(("stop", "rm")) for c in docker_calls(work))


def test_script_is_bash3_compatible() -> None:
    """macOS ships bash 3.2: no bash 4+ constructs in ./lainbow."""
    text = (ROOT / "lainbow").read_text()
    banned = [
        r"declare\s+-[a-zA-Z]*[Aun]\b",
        r"\bmapfile\b",
        r"\breadarray\b",
        r"\$\{[A-Za-z_0-9]+(,,|\^\^|,|\^)[^}]*\}",
        r"\[\[\s+-v\b",
        r";&",
        r"\|&",
        r"%\([^)]*\)T",
        r"\$\{[A-Za-z_0-9]+@[UuLlQEPAaK]\}",
    ]
    for pattern in banned:
        found = [ln for ln in text.splitlines() if re.search(pattern, ln)]
        assert not found, f"bash 4+ construct {pattern!r}: {found}"


def test_script_parses_in_posix_mode() -> None:
    result = subprocess.run(
        ["bash", "--posix", "-n", str(ROOT / "lainbow")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_status_and_update_call_compose(work: Path) -> None:
    assert run(work, "status").returncode == 0
    assert run(work, "config").returncode == 0
    install_fake(work, "git", FAKE_GIT)
    result = run(work, "update")
    assert result.returncode == 0, result.stderr
    assert "git checkout <tag> && ./lainbow update" in result.stdout
    assert "v2.1.0" in result.stdout
    git_log = (work.parent / "git.log").read_text().splitlines()
    assert git_log[0] == "fetch --tags"
    assert not any(c.startswith(("checkout", "pull", "reset")) for c in git_log)
    log = (work.parent / "docker.log").read_text()
    assert "lainbow-admin status" in log
    assert "lainbow-admin config" in log
    assert "compose pull" not in log
    assert "compose up -d --build" in log
    assert " -p " not in log


def test_help_and_unknown(work: Path) -> None:
    out = run(work).stdout
    assert "add-engine" in out
    assert "join" in out
    assert "leave" in out
    assert run(work, "bogus").returncode != 0

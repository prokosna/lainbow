"""WI-M4-02: Release tag detection and LAINBOW_VERSION handling in `./lainbow setup`.

Runs `setup` inside temporary git repositories to verify that:
1. At a release tag (`v*.*.*`), setup sets `LAINBOW_VERSION=<tag without v>`.
2. Without a release tag, setup does not set `LAINBOW_VERSION` (local build mode).
3. If `LAINBOW_VERSION` already exists in `.env`, setup preserves it.
"""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAINBOW_SRC = REPO_ROOT / "lainbow"

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "Test Author",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test Committer",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def _values(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in env_file.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key, value = stripped.split("=", 1)
            values[key] = value
    return values


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    lainbow_dest = repo / "lainbow"
    shutil.copy2(LAINBOW_SRC, lainbow_dest)
    lainbow_dest.chmod(lainbow_dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    subprocess.run(["git", "init"], cwd=repo, env=GIT_ENV, check=True, capture_output=True)
    subprocess.run(
        ["git", "add", "lainbow"], cwd=repo, env=GIT_ENV, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo,
        env=GIT_ENV,
        check=True,
        capture_output=True,
    )
    return repo


@pytest.fixture
def music_dir(tmp_path: Path) -> Path:
    path = tmp_path / "music"
    path.mkdir()
    return path


def _setup(repo_dir: Path, env_file: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **GIT_ENV,
        "LAINBOW_ENV_FILE": str(env_file),
        "LAINBOW_SKIP_PREREQS": "1",
    }
    return subprocess.run(
        [str(repo_dir / "lainbow"), "setup", "--yes", "--no-download", *args],
        cwd=repo_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_tag_appends_version_on_fresh_setup(
    git_repo: Path, music_dir: Path, tmp_path: Path
) -> None:
    subprocess.run(
        ["git", "tag", "v9.9.9"],
        cwd=git_repo,
        env=GIT_ENV,
        check=True,
        capture_output=True,
    )
    env_file = tmp_path / ".env"

    result = _setup(git_repo, env_file, "--music-dir", str(music_dir), "--accel", "cpu")

    assert result.returncode == 0, result.stderr
    values = _values(env_file)
    assert values["LAINBOW_VERSION"] == "9.9.9"
    assert "Mode: prebuilt images (9.9.9)" in result.stdout


def test_tag_appends_version_on_existing_env(
    git_repo: Path, music_dir: Path, tmp_path: Path
) -> None:
    subprocess.run(
        ["git", "tag", "v9.9.9"],
        cwd=git_repo,
        env=GIT_ENV,
        check=True,
        capture_output=True,
    )
    env_file = tmp_path / ".env"
    env_file.write_text(f"MUSIC_NAS_ROOT_DIR={music_dir}\n")

    result = _setup(git_repo, env_file, "--accel", "cpu")

    assert result.returncode == 0, result.stderr
    values = _values(env_file)
    assert values["LAINBOW_VERSION"] == "9.9.9"
    assert "Mode: prebuilt images (9.9.9)" in result.stdout


def test_no_tag_appends_no_version(git_repo: Path, music_dir: Path, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"

    result = _setup(git_repo, env_file, "--music-dir", str(music_dir), "--accel", "cpu")

    assert result.returncode == 0, result.stderr
    values = _values(env_file)
    assert "LAINBOW_VERSION" not in values
    assert "Mode: local build" in result.stdout


def test_existing_version_is_preserved(git_repo: Path, music_dir: Path, tmp_path: Path) -> None:
    subprocess.run(
        ["git", "tag", "v9.9.9"],
        cwd=git_repo,
        env=GIT_ENV,
        check=True,
        capture_output=True,
    )
    env_file = tmp_path / ".env"
    env_file.write_text(f"MUSIC_NAS_ROOT_DIR={music_dir}\nLAINBOW_VERSION=1.0.0\n")

    result = _setup(git_repo, env_file, "--accel", "cpu")

    assert result.returncode == 0, result.stderr
    values = _values(env_file)
    assert values["LAINBOW_VERSION"] == "1.0.0"
    assert "Mode: prebuilt images (1.0.0)" in result.stdout


def test_help_shows_current_mode(git_repo: Path, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LAINBOW_VERSION=2.0.0\n")

    env = {
        **GIT_ENV,
        "LAINBOW_ENV_FILE": str(env_file),
    }
    result = subprocess.run(
        [str(git_repo / "lainbow"), "help"],
        cwd=git_repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0
    assert "Current mode: prebuilt images (2.0.0)" in result.stdout

    # Without env file (defaults to local)
    result_default = subprocess.run(
        [str(git_repo / "lainbow"), "help"],
        cwd=git_repo,
        env=GIT_ENV,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result_default.returncode == 0
    assert "Current mode: local build" in result_default.stdout


@pytest.fixture
def versioned_remote_repo(tmp_path: Path) -> Path:
    """Creates a client repository with a bare origin remote.

    Tags v1.0.0 and v1.1.0 exist on remote; client checkout is at v1.0.0 with
    a tracked file committed at v1.0.0.
    """
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)

    client = tmp_path / "client"
    client.mkdir()
    lainbow_dest = client / "lainbow"
    shutil.copy2(LAINBOW_SRC, lainbow_dest)
    lainbow_dest.chmod(lainbow_dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    tracked = client / "tracked.txt"
    tracked.write_text("v1.0.0\n")

    subprocess.run(["git", "init"], cwd=client, env=GIT_ENV, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", str(bare)],
        cwd=client,
        env=GIT_ENV,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "lainbow", "tracked.txt"],
        cwd=client,
        env=GIT_ENV,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "v1.0.0"], cwd=client, env=GIT_ENV, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "tag", "v1.0.0"], cwd=client, env=GIT_ENV, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "push", "origin", "HEAD", "--tags"],
        cwd=client,
        env=GIT_ENV,
        check=True,
        capture_output=True,
    )

    tracked.write_text("v1.1.0\n")
    subprocess.run(
        ["git", "add", "tracked.txt"], cwd=client, env=GIT_ENV, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "v1.1.0"], cwd=client, env=GIT_ENV, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "tag", "v1.1.0"], cwd=client, env=GIT_ENV, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "push", "origin", "HEAD", "--tags"],
        cwd=client,
        env=GIT_ENV,
        check=True,
        capture_output=True,
    )

    # Return client to v1.0.0 and delete local tag v1.1.0 to test remote fetch
    subprocess.run(
        ["git", "checkout", "-q", "v1.0.0"],
        cwd=client,
        env=GIT_ENV,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "tag", "-d", "v1.1.0"], cwd=client, env=GIT_ENV, check=True, capture_output=True
    )

    return client


def _update(
    repo_dir: Path, *args: str, input: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = {
        **GIT_ENV,
        "LAINBOW_SKIP_PREREQS": "1",
    }
    return subprocess.run(
        [str(repo_dir / "lainbow"), "update", *args],
        cwd=repo_dir,
        env=env,
        input=input,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_update_prebuilt_advances_checkout_and_updates_env(
    versioned_remote_repo: Path,
) -> None:
    client = versioned_remote_repo
    env_file = client / ".env"
    initial_content = (
        "# Configuration\nMUSIC_NAS_ROOT_DIR=/music\nLAINBOW_VERSION=1.0.0\nWEB_API_PORT=8000\n"
    )
    env_file.write_text(initial_content)

    result = _update(client, "--yes", "--no-start")

    assert result.returncode == 0, result.stderr
    assert "1.0.0 -> 1.1.0" in result.stdout

    # Checkout is at v1.1.0
    current_tag = subprocess.run(
        ["git", "describe", "--tags", "--exact-match"],
        cwd=client,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert current_tag == "v1.1.0"

    # .env has LAINBOW_VERSION=1.1.0 with every other line unchanged
    expected_content = (
        "# Configuration\nMUSIC_NAS_ROOT_DIR=/music\nLAINBOW_VERSION=1.1.0\nWEB_API_PORT=8000\n"
    )
    assert env_file.read_text() == expected_content

    # A backup exists with the original content
    backups = list(client.glob(".env.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_text() == initial_content


def test_update_already_at_newest_tag_changes_nothing(
    versioned_remote_repo: Path,
) -> None:
    client = versioned_remote_repo
    env_file = client / ".env"
    env_file.write_text("LAINBOW_VERSION=1.0.0\n")

    # Update to 1.1.0 first
    assert _update(client, "--yes", "--no-start").returncode == 0
    backups_before = list(client.glob(".env.bak.*"))
    assert len(backups_before) == 1
    content_before = env_file.read_text()

    # Second update when already at newest
    result = _update(client, "--yes", "--no-start")

    assert result.returncode == 0, result.stderr
    assert "Already at 1.1.0" in result.stdout

    # Nothing changed and no new backup made
    current_tag = subprocess.run(
        ["git", "describe", "--tags", "--exact-match"],
        cwd=client,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert current_tag == "v1.1.0"
    assert env_file.read_text() == content_before
    assert list(client.glob(".env.bak.*")) == backups_before


def test_update_refuses_when_tracked_file_is_dirty(
    versioned_remote_repo: Path,
) -> None:
    client = versioned_remote_repo
    env_file = client / ".env"
    env_content = "LAINBOW_VERSION=1.0.0\n"
    env_file.write_text(env_content)

    # Modify a tracked file
    (client / "tracked.txt").write_text("uncommitted dirty changes\n")

    result = _update(client, "--yes", "--no-start")

    assert result.returncode != 0
    assert "uncommitted changes" in result.stderr

    # Checkout and .env unchanged, no backup created
    current_tag = subprocess.run(
        ["git", "describe", "--tags", "--exact-match"],
        cwd=client,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert current_tag == "v1.0.0"
    assert env_file.read_text() == env_content
    assert list(client.glob(".env.bak.*")) == []


def test_update_aborted_when_user_declines_confirmation(
    versioned_remote_repo: Path,
) -> None:
    client = versioned_remote_repo
    env_file = client / ".env"
    env_content = "LAINBOW_VERSION=1.0.0\n"
    env_file.write_text(env_content)

    result = _update(client, "--no-start", input="n\n")

    assert result.returncode == 0
    assert "1.0.0 -> 1.1.0" in result.stdout
    assert "Update aborted." in result.stdout

    current_tag = subprocess.run(
        ["git", "describe", "--tags", "--exact-match"],
        cwd=client,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert current_tag == "v1.0.0"
    assert env_file.read_text() == env_content
    assert list(client.glob(".env.bak.*")) == []


def test_update_help(versioned_remote_repo: Path) -> None:
    result = _update(versioned_remote_repo, "--help")
    assert result.returncode == 0
    assert "Usage: ./lainbow update" in result.stdout
    assert "--yes" in result.stdout
    assert "--no-start" in result.stdout

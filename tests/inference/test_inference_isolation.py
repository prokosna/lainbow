import os
import subprocess
import sys
from pathlib import Path


def test_inference_isolation() -> None:
    """Verifies that inference modules do not import sqlmodel or sqlalchemy."""
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src{os.pathsep}{repo_root / 'src'}"

    code = (
        "import sys\n"
        "sys.modules['sqlmodel'] = None\n"
        "sys.modules['sqlalchemy'] = None\n"
        "import inference.main\n"
        "import inference.download_models\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"Subprocess failed with code {result.returncode}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

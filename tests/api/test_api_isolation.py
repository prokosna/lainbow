import os
import subprocess
import sys


def test_api_isolation() -> None:
    """Assert that importing api.main does not load heavy dependencies."""
    code = (
        "import sys\n"
        "import api.main\n"
        "heavy = ['torch', 'librosa', 'laion_clap', 'muq', 'mutagen']\n"
        "imported = [m for m in heavy if m in sys.modules]\n"
        "if imported:\n"
        "    raise AssertionError(f'Heavy modules imported: {imported}')\n"
    )
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"src:{current_pythonpath}" if current_pythonpath else "src"

    res = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

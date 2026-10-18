#!/bin/sh
# Runs every gate (AGENTS.md section 3-3) regardless of earlier failures, so one
# failure never hides another. Needs the dev environment:
#   uv sync --group dev --group api --group batch --group inference --group inference-cpu
set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR" || exit 1

# mypy may not get worse than this; lower it as errors are fixed, never raise it.
MYPY_BASELINE_FILE="$SCRIPT_DIR/mypy-baseline.txt"

if [ ! -d .venv ]; then
	echo "no .venv: run 'uv sync --group dev --group api --group batch --group inference --group inference-cpu'" >&2
	exit 1
fi

failed=""

run() {
	name="$1"
	shift
	echo "=== $name"
	if ! "$@"; then
		failed="$failed $name"
	fi
}

mypy_gate() {
	baseline=$(cat "$MYPY_BASELINE_FILE")
	output=$(uv run --no-sync mypy src 2>&1)
	count=$(printf '%s\n' "$output" | grep -c ': error:')
	echo "mypy errors: $count (baseline $baseline)"
	if [ "$count" -gt "$baseline" ]; then
		printf '%s\n' "$output" | grep ': error:'
		echo "mypy got worse than the baseline" >&2
		return 1
	fi
	if [ "$count" -lt "$baseline" ]; then
		echo "mypy improved: lower $MYPY_BASELINE_FILE to $count"
	fi
	return 0
}

shellcheck_gate() {
	uvx --from shellcheck-py shellcheck scripts/*.sh lainbow
}

run ruff-check uv run --no-sync ruff check src tests
run ruff-format uv run --no-sync ruff format --check src tests
run mypy mypy_gate
run pytest uv run --no-sync pytest -q

# Tests marked postgres need a disposable PostgreSQL; they run only when one is given.
if [ -n "${LAINBOW_TEST_POSTGRES_URL:-}" ]; then
	run pytest-postgres uv run --no-sync pytest -q -m postgres
else
	echo "=== pytest-postgres: skipped (set LAINBOW_TEST_POSTGRES_URL to run)"
fi
run shellcheck shellcheck_gate

echo "==="
if [ -n "$failed" ]; then
	echo "FAILED:$failed"
	exit 1
fi
echo "all gates passed"

#!/bin/sh
# Runs every gate regardless of earlier failures, so one failure never hides another.
# Needs the dev environment: uv sync --extra server --extra engine --extra models --extra optional-models --extra cpu
set -u

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR" || exit 1

if [ ! -d .venv ]; then
	echo "no .venv: run 'uv sync --extra server --extra engine --extra models --extra optional-models --extra cpu'" >&2
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

shell_files() {
	find scripts -name '*.sh'
	[ -f lainbow ] && echo lainbow
}

run ruff uv run --no-sync ruff check .
run ruff-format uv run --no-sync ruff format --check .
run mypy uv run --no-sync mypy
run import-linter uv run --no-sync lint-imports
run pytest uv run --no-sync pytest -q
# The shellcheck-py wheel ships a static binary, so no system package is needed
# shellcheck disable=SC2046
run shellcheck uvx --from shellcheck-py shellcheck $(shell_files)

echo "==="
if [ -n "$failed" ]; then
	echo "failed:$failed" >&2
	exit 1
fi
echo "all gates passed"

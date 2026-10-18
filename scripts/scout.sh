#!/bin/sh
# Asks a cheap model a read-only question about the repository (AGENTS.md section 3-2).
# The answer is a hypothesis: verify any file:line it cites before relying on it.
set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

MODEL=${LAINBOW_SCOUT_MODEL:-gemini-3.8-flash-low}
PRINT_TIMEOUT=${LAINBOW_SCOUT_TIMEOUT:-10m}

if [ $# -lt 1 ]; then
	echo "usage: scripts/scout.sh \"<question>\"" >&2
	echo "  env: LAINBOW_SCOUT_MODEL   (default $MODEL)" >&2
	echo "       LAINBOW_SCOUT_TIMEOUT (default $PRINT_TIMEOUT)" >&2
	exit 2
fi

if ! command -v agy > /dev/null 2>&1; then
	echo "agy is not on PATH" >&2
	exit 1
fi

cd "$ROOT_DIR" || exit 1

PREAMBLE="You are a read-only scout for the repository in the current workspace.
Rules: do not create, edit, move or delete any file; do not run git commands that change state.
Answer concisely, in the language of the question, and cite evidence as path:line for every claim.
If you are not sure, say so instead of guessing.

Question:"

tree_fingerprint() {
	git status --porcelain
	git diff HEAD --binary
	git ls-files --others --exclude-standard | git hash-object --stdin-paths
}

tree_before=$(tree_fingerprint)
out=$(mktemp)
trap 'rm -f "$out"' EXIT

# plan mode keeps the agent from applying edits; the tree check below enforces it regardless.
agy --model "$MODEL" \
	--mode plan \
	--print-timeout="$PRINT_TIMEOUT" \
	--add-dir "$ROOT_DIR" \
	--print="$PREAMBLE $*" > "$out" 2>&1
status=$?

cat "$out"

if [ "$tree_before" != "$(tree_fingerprint)" ]; then
	echo "SCOUT MODIFIED THE TREE: inspect git status and revert its changes" >&2
	exit 1
fi

if grep -Eq "timeout waiting for response|print timeout after" "$out"; then
	echo "SCOUT WAS CUT OFF by agy's print timeout ($PRINT_TIMEOUT): the answer is incomplete" >&2
	exit 1
fi

exit $status

#!/bin/sh
# Dispatches a Work Order to the out-of-session agy Implementer (AGENTS.md section 3-1).
# Ported from tradr's ci/dispatch-implementer.sh; the failure checks below exist
# because agy exits 0 in cases where the run did not actually complete.
set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

# agy's own --print-timeout default is too short for a Work Order that runs the gates.
PRINT_TIMEOUT=${LAINBOW_PRINT_TIMEOUT:-30m}
MODEL=${LAINBOW_IMPLEMENTER_MODEL:-gemini-3.8-flash-high}

if [ $# -lt 1 ]; then
	echo "usage: scripts/dispatch-implementer.sh <work-order-file> [log-file]" >&2
	echo "  env: LAINBOW_IMPLEMENTER_MODEL (default $MODEL)" >&2
	echo "       LAINBOW_PRINT_TIMEOUT     (default $PRINT_TIMEOUT)" >&2
	exit 2
fi

ORDER="$1"
LOG=${2:-"$ROOT_DIR/.dispatch-$(basename "$ORDER" .md).log"}

if [ ! -f "$ORDER" ]; then
	echo "$ORDER: work order not found" >&2
	exit 1
fi

if ! command -v agy > /dev/null 2>&1; then
	echo "agy is not on PATH" >&2
	exit 1
fi

cd "$ROOT_DIR" || exit 1

# A REVISE round starts from a dirty tree, so compare content rather than the file list.
tree_fingerprint() {
	git status --porcelain
	git diff HEAD --binary
	git ls-files --others --exclude-standard | git hash-object --stdin-paths
}

# Comparing against origin/main keeps a mid-run merge from reading as an Implementer commit.
head_before=$(git rev-parse HEAD)
tree_before=$(tree_fingerprint)
has_origin_main=1
commits_before=""
if ! git rev-parse --verify origin/main > /dev/null 2>&1; then
	echo "dispatch-implementer: origin/main does not resolve, falling back to HEAD comparison" >&2
	has_origin_main=0
else
	commits_before=$(git log --format=%H HEAD --not origin/main)
fi

echo "dispatching $ORDER"
echo "  model         $MODEL"
echo "  print-timeout $PRINT_TIMEOUT"
echo "  log           $LOG"
echo "  HEAD before   $head_before"

agy --model "$MODEL" \
	--print-timeout="$PRINT_TIMEOUT" \
	--add-dir "$ROOT_DIR" \
	--print="$(cat "$ORDER")" > "$LOG" 2>&1
status=$?

head_after=$(git rev-parse HEAD)
tree_after=$(tree_fingerprint)

echo "  HEAD after    $head_after"
echo "  exit          $status"

if [ "$has_origin_main" -eq 1 ]; then
	commits_after=$(git log --format=%H HEAD --not origin/main)
	new_commits=""
	for c in $commits_after; do
		found=0
		for b in $commits_before; do
			if [ "$c" = "$b" ]; then
				found=1
				break
			fi
		done
		if [ "$found" -eq 0 ]; then
			new_commits="$new_commits $c"
		fi
	done
	if [ -n "$new_commits" ]; then
		echo "IMPLEMENTER COMMITTED (AGENTS.md section 4):" >&2
		for c in $new_commits; do
			git log --oneline -1 "$c" >&2
		done
		exit 1
	fi
elif [ "$head_before" != "$head_after" ]; then
	echo "IMPLEMENTER COMMITTED: HEAD moved $head_before -> $head_after (AGENTS.md section 4)" >&2
	exit 1
fi

# agy writes its own cut-off into the log and still exits 0.
if grep -Eq "timeout waiting for response|print timeout after" "$LOG" 2>/dev/null; then
	echo "RUN WAS CUT OFF by agy's print timeout ($PRINT_TIMEOUT): the report is incomplete" >&2
	echo "raise LAINBOW_PRINT_TIMEOUT and dispatch again, or review the tree directly" >&2
	exit 1
fi

# An agent idling on a background task is terminated with exit status 0.
if grep -Eq "terminating .* background task" "$LOG" 2>/dev/null; then
	echo "RUN WAS TERMINATED with a background task still running: the report is incomplete" >&2
	if [ "$tree_before" = "$tree_after" ]; then
		echo "tree unchanged: the run produced no work, dispatch it again" >&2
	else
		echo "tree changed: review the tree and run every gate here" >&2
	fi
	exit 1
fi

exit $status

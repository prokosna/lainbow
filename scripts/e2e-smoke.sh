#!/bin/sh
# End-to-end smoke test for Lainbow.
# Automated flow: throwaway compose stack -> audio generation -> scan -> analyze (acoustic_features) -> similar & stats.
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")
cd "$REPO_ROOT" || exit 1

PROJECT_NAME="lainbow-e2e-$$"
E2E_PORT_BASE="${E2E_PORT_BASE:-20000}"

WEB_API_PORT=$((E2E_PORT_BASE + 0))
POSTGRES_PORT=$((E2E_PORT_BASE + 1))
RABBITMQ_PORT=$((E2E_PORT_BASE + 2))
RABBITMQ_MANAGEMENT_PORT=$((E2E_PORT_BASE + 3))
QDRANT_PORT_HTTP=$((E2E_PORT_BASE + 4))
QDRANT_PORT_GRPC=$((E2E_PORT_BASE + 5))
INFERENCE_API_PORT=$((E2E_PORT_BASE + 6))

WORK_DIR=""
ENV_FILE=""

# shellcheck disable=SC2329
cleanup() {
	exit_code=$?
	trap - EXIT INT TERM HUP
	echo "=== Cleaning up smoke test environment ($PROJECT_NAME)..."
	if [ -n "$WORK_DIR" ] && [ -f "$ENV_FILE" ]; then
		docker compose -p "$PROJECT_NAME" --env-file "$ENV_FILE" down -v --remove-orphans >/dev/null 2>&1 || true
	fi
	if [ -n "$WORK_DIR" ] && [ -d "$WORK_DIR" ]; then
		if ! rm -rf "$WORK_DIR" 2>/dev/null; then
			docker run --rm -v "$WORK_DIR":/d alpine rm -rf /d/data >/dev/null 2>&1 || \
				docker run --rm --entrypoint "" -v "$WORK_DIR":/d postgres:17.5-alpine rm -rf /d/data >/dev/null 2>&1 || true
			rm -rf "$WORK_DIR" 2>/dev/null || true
		fi
	fi
	exit "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

get_json_field() {
	raw_json="$1"
	target_field="$2"
	if command -v python3 >/dev/null 2>&1; then
		printf '%s' "$raw_json" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get(sys.argv[1], ""))
except Exception:
    pass
' "$target_field" 2>/dev/null || true
	else
		case "$target_field" in
			id|status|name)
				printf '%s' "$raw_json" | grep -o "\"$target_field\": *\"[^\"]*\"" | head -n 1 | cut -d: -f2 | tr -d ' "' || true
				;;
			progress|total_songs|songs_with_acoustic_features)
				printf '%s' "$raw_json" | grep -o "\"$target_field\": *[0-9]*" | head -n 1 | cut -d: -f2 | tr -d ' ' || true
				;;
			message)
				printf '%s' "$raw_json" | sed -n 's/.*"message"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' || true
				;;
			*)
				echo ""
				;;
		esac
	fi
}

poll_task() {
	target_task_id="$1"
	task_desc="$2"
	poll_elapsed=0
	poll_timeout=300
	poll_last_status=""
	poll_last_progress=""
	task_info=""

	echo "Waiting for $task_desc (task ID: $target_task_id, max ${poll_timeout}s)..."
	while [ "$poll_elapsed" -lt "$poll_timeout" ]; do
		task_info=$(curl -s -S "http://127.0.0.1:${WEB_API_PORT}/api/v1/batch/tasks/${target_task_id}" 2>/dev/null || true)
		if [ -z "$task_info" ]; then
			sleep 2
			poll_elapsed=$((poll_elapsed + 2))
			continue
		fi

		cur_status=$(get_json_field "$task_info" status)
		cur_progress=$(get_json_field "$task_info" progress)
		cur_message=$(get_json_field "$task_info" message)

		if [ "$cur_status" != "$poll_last_status" ] || [ "$cur_progress" != "$poll_last_progress" ]; then
			printf '  [%s] Status: %s | Progress: %s%%' "$task_desc" "${cur_status:-UNKNOWN}" "${cur_progress:-0}"
			if [ -n "$cur_message" ]; then
				printf ' | %s' "$cur_message"
			fi
			printf '\n'
			poll_last_status="$cur_status"
			poll_last_progress="$cur_progress"
		fi

		case "$cur_status" in
			SUCCESS)
				echo "  $task_desc completed successfully."
				return 0
				;;
			FAILURE)
				echo "Error: $task_desc failed! API response: $task_info" >&2
				return 1
				;;
			SKIPPED)
				echo "  $task_desc was skipped."
				return 0
				;;
		esac

		sleep 2
		poll_elapsed=$((poll_elapsed + 2))
	done

	echo "Error: $task_desc timed out after ${poll_timeout}s! Last response: $task_info" >&2
	return 1
}

echo "=== [1/7] Preparing temporary environment in work directory..."
WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/lainbow-e2e.XXXXXX")
DATA_DIR="$WORK_DIR/data"
MUSIC_DIR="$WORK_DIR/music"
MODELS_DIR="$WORK_DIR/models"
ENV_FILE="$WORK_DIR/.env"

mkdir -p "$DATA_DIR" "$MUSIC_DIR" "$MODELS_DIR"

POSTGRES_PASSWORD=$(LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 24)
RABBITMQ_DEFAULT_PASS=$(LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 24)

cat > "$ENV_FILE" <<EOF
LAINBOW_PROJECT_NAME=$PROJECT_NAME
COMPOSE_PROFILES=db,app
DATABASE_DATA_ROOT_DIR=$DATA_DIR
MUSIC_NAS_ROOT_DIR=$MUSIC_DIR
LAINBOW_MODELS_DIR=$MODELS_DIR
POSTGRES_USER=user
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
POSTGRES_DB=lainbow_test_db
RABBITMQ_DEFAULT_USER=user
RABBITMQ_DEFAULT_PASS=$RABBITMQ_DEFAULT_PASS
LAINBOW_BIND_ADDRESS=127.0.0.1
WEB_API_PORT=$WEB_API_PORT
POSTGRES_PORT=$POSTGRES_PORT
RABBITMQ_PORT=$RABBITMQ_PORT
RABBITMQ_MANAGEMENT_PORT=$RABBITMQ_MANAGEMENT_PORT
QDRANT_PORT_HTTP=$QDRANT_PORT_HTTP
QDRANT_PORT_GRPC=$QDRANT_PORT_GRPC
INFERENCE_API_PORT=$INFERENCE_API_PORT
BATCH_CONCURRENCY=1
EOF

echo "=== [2/7] Building images (api, batch)..."
docker compose -p "$PROJECT_NAME" --env-file "$ENV_FILE" --profile db --profile app build

echo "=== [3/7] Starting services (profiles: db, app)..."
docker compose -p "$PROJECT_NAME" --env-file "$ENV_FILE" --profile db --profile app up -d

echo "=== [4/7] Generating synthetic audio files in batch container..."
docker compose -p "$PROJECT_NAME" --env-file "$ENV_FILE" run --rm --no-deps -v "$MUSIC_DIR":/out batch python -c \
	'import numpy as np, soundfile as sf; sr = 22050; t = np.linspace(0, 3, int(sr * 3), endpoint=False); [sf.write(f"/out/tone_{i+1}.wav", 0.5 * np.sin(2 * np.pi * freq * t), sr) for i, freq in enumerate([440, 880, 1760])]'

echo "=== [5/7] Waiting for API to become healthy at http://127.0.0.1:${WEB_API_PORT}/health (max 180s)..."
api_healthy=false
health_elapsed=0
health_timeout=180
health_last_resp=""
while [ "$health_elapsed" -lt "$health_timeout" ]; do
	health_resp=$(curl -s -S "http://127.0.0.1:${WEB_API_PORT}/health" 2>&1 || true)
	health_last_resp="$health_resp"
	if echo "$health_resp" | grep -q '"status"'; then
		echo "API is healthy: $health_resp"
		api_healthy=true
		break
	fi
	sleep 2
	health_elapsed=$((health_elapsed + 2))
done

if [ "$api_healthy" != true ]; then
	echo "Error: Timed out waiting for API /health after ${health_timeout}s. Last response: $health_last_resp" >&2
	docker compose -p "$PROJECT_NAME" --env-file "$ENV_FILE" logs --tail 50 >&2 || true
	exit 1
fi

echo "=== [6/7] Running scan and analysis tasks..."
echo "Starting library scan..."
scan_resp=$(curl -s -S -X POST "http://127.0.0.1:${WEB_API_PORT}/api/v1/batch/scan/run" 2>&1 || true)
scan_task_id=$(get_json_field "$scan_resp" id)
if [ -z "$scan_task_id" ]; then
	echo "Error: Failed to enqueue scan task. API response: $scan_resp" >&2
	exit 1
fi
echo "Scan task submitted: $scan_task_id"
poll_task "$scan_task_id" "Library scan" || exit 1

echo "Starting acoustic features analysis..."
analyze_resp=$(curl -s -S -X POST "http://127.0.0.1:${WEB_API_PORT}/api/v1/batch/analyze/run" \
	-H "Content-Type: application/json" \
	-d '{"models":["acoustic_features"]}' 2>&1 || true)
analyze_task_id=$(get_json_field "$analyze_resp" id)
if [ -z "$analyze_task_id" ]; then
	echo "Error: Failed to enqueue analyze task. API response: $analyze_resp" >&2
	exit 1
fi
echo "Analyze task submitted: $analyze_task_id"
poll_task "$analyze_task_id" "Acoustic features analysis" || exit 1

echo "=== [7/7] Verifying similarity search and database statistics..."
echo "Checking /api/v1/songs/tone_1.wav/similar?model_name=acoustic_features..."
similar_resp=$(curl -s -S "http://127.0.0.1:${WEB_API_PORT}/api/v1/songs/tone_1.wav/similar?model_name=acoustic_features" 2>&1 || true)

has_similar=false
if command -v python3 >/dev/null 2>&1; then
	if printf '%s' "$similar_resp" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
    sys.exit(0 if len(data.get("songs", [])) > 0 else 1)
except Exception:
    sys.exit(1)
' 2>/dev/null; then
		has_similar=true
	fi
else
	if printf '%s' "$similar_resp" | grep -q '"file_path"'; then
		has_similar=true
	fi
fi

if [ "$has_similar" != true ]; then
	echo "Error: Similarity search did not return any songs! API response: $similar_resp" >&2
	exit 1
fi
echo "Similarity search verified (found similar songs)."

echo "Checking /api/v1/stats..."
stats_resp=$(curl -s -S "http://127.0.0.1:${WEB_API_PORT}/api/v1/stats" 2>&1 || true)

stats_ok=false
if command -v python3 >/dev/null 2>&1; then
	if printf '%s' "$stats_resp" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
    total = data.get("total_songs", 0)
    acoustic = data.get("songs_with_acoustic_features", 0)
    sys.exit(0 if total > 0 and total == acoustic else 1)
except Exception:
    sys.exit(1)
' 2>/dev/null; then
		stats_ok=true
	fi
else
	total_songs=$(get_json_field "$stats_resp" total_songs)
	acoustic_songs=$(get_json_field "$stats_resp" songs_with_acoustic_features)
	if [ -n "$total_songs" ] && [ "$total_songs" -gt 0 ] 2>/dev/null && [ "$total_songs" = "$acoustic_songs" ]; then
		stats_ok=true
	fi
fi

if [ "$stats_ok" != true ]; then
	echo "Error: Stats check failed! Expected all songs to have acoustic features. API response: $stats_resp" >&2
	exit 1
fi
echo "Database statistics verified (all songs have acoustic features): $stats_resp"

echo "=== All smoke tests passed successfully ==="
exit 0

#!/bin/sh
# End-to-end smoke test of the container images and compose.yaml.
# It starts postgres and the server (no engine: real models need a multi-GB download), checks the startup scan
# and the engine API, and proves that the engine image imports torch and MuQ.
# Everything runs in the compose project lainbow-v2-e2e and is removed on exit.
set -eu

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PROJECT=lainbow-v2-e2e
PORT=18000
TOKEN=lbe_00000000000000000000000000000001_e2e-secret
ENGINE_IMAGE=lainbow-engine:e2e-cpu

WORK_DIR=$(mktemp -d)
COMPOSE_STARTED=""

fail() {
	echo "e2e: FAIL: $*" >&2
	exit 1
}

dc() {
	docker compose -p "$PROJECT" -f "$WORK_DIR/compose.yaml" -f "$WORK_DIR/override.yaml" \
		--project-directory "$WORK_DIR" "$@"
}

cleanup() {
	status=$?
	if [ -n "$COMPOSE_STARTED" ]; then
		if [ "$status" -ne 0 ]; then
			dc logs --no-color --tail 50 >&2 || true
		fi
		dc down -v --remove-orphans >/dev/null 2>&1 || true
	fi
	# Postgres files are owned by another uid, so remove them from inside a container.
	docker run --rm -v "$WORK_DIR:/work" --entrypoint rm "$ENGINE_IMAGE" -rf /work/data >/dev/null 2>&1 || true
	rm -rf "$WORK_DIR"
	exit "$status"
}
trap cleanup EXIT
trap 'exit 1' INT TERM

mkdir -p "$WORK_DIR/music/album" "$WORK_DIR/data"
cp "$ROOT_DIR/compose.yaml" "$WORK_DIR/compose.yaml"
cat >"$WORK_DIR/.env" <<EOT
MUSIC_DIR=$WORK_DIR/music
DATA_DIR=$WORK_DIR/data
PORT=$PORT
COMPOSE_PROFILES=cpu
POSTGRES_PASSWORD=e2e-password
ENGINE_TOKEN=$TOKEN
LAINBOW_VERSION=e2e
EOT
# Builds from the checkout (the copied compose.yaml lives elsewhere) and tags the images lainbow-*:e2e.
cat >"$WORK_DIR/override.yaml" <<EOT
services:
  server:
    build:
      context: $ROOT_DIR
  engine-cpu:
    build:
      context: $ROOT_DIR
EOT

echo "e2e: building images"
COMPOSE_STARTED=1
dc build server engine-cpu || fail "image build failed"

echo "e2e: generating test audio with ffmpeg from the engine image"
docker run --rm -v "$WORK_DIR/music:/music" --entrypoint sh "$ENGINE_IMAGE" -c '
	set -e
	for f in 220 440 880; do
		ffmpeg -loglevel error -f lavfi -i "sine=frequency=$f:duration=2" "/music/album/tone$f.wav"
	done
' || fail "could not generate test audio"
[ "$(find "$WORK_DIR/music" -name '*.wav' | wc -l)" -eq 3 ] || fail "expected 3 test WAV files"

echo "e2e: starting postgres and server"
dc up -d postgres server || fail "could not start postgres and server"

BASE="http://127.0.0.1:$PORT"

wait_for() {
	# usage: wait_for <description> <timeout-sec> <command...>
	desc=$1
	limit=$2
	shift 2
	waited=0
	until "$@" >/dev/null 2>&1; do
		waited=$((waited + 2))
		[ "$waited" -le "$limit" ] || fail "timed out waiting for $desc"
		sleep 2
	done
}

stats_has_three_songs() {
	curl -fsS "$BASE/api/v1/stats" | grep -Eq '"total_songs": *3[^0-9]'
}

echo "e2e: waiting for /health"
wait_for "/health" 120 curl -fsS "$BASE/health"

echo "e2e: waiting for the startup scan to find 3 songs"
wait_for "total_songs=3 in /api/v1/stats" 120 stats_has_three_songs

echo "e2e: checking engine API authentication"
code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/engine/v1/plan")
[ "$code" = 401 ] || fail "/engine/v1/plan without a token returned $code, expected 401"
code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOKEN" "$BASE/engine/v1/plan")
[ "$code" = 200 ] || fail "/engine/v1/plan with the token returned $code, expected 200"

echo "e2e: checking that the engine image imports torch and MuQ"
docker run --rm --entrypoint python "$ENGINE_IMAGE" \
	-c 'import torch, lainbow.engine.models.muq; print("torch", torch.__version__)' ||
	fail "the engine image cannot import torch and muq"

echo "e2e: image sizes"
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' "lainbow-*:e2e*"

echo "e2e: PASS"

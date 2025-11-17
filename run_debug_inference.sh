#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model_name> <audio_path> [--pretty] [extra args...]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

MODEL_NAME="$1"
AUDIO_PATH="$2"
shift 2

if ! ABS_AUDIO_PATH="$(realpath -e "$AUDIO_PATH" 2>/dev/null)"; then
  echo "Audio file not found: $AUDIO_PATH" >&2
  exit 1
fi

AUDIO_DIR="$(dirname "$ABS_AUDIO_PATH")"

RUNTIME="${INFERENCE_RUNTIME:-cuda}"

GPU_ARGS=()
IMAGE_NAME="${INFERENCE_DOCKER_IMAGE:-}"

case "$RUNTIME" in
  cuda)
    GPU_ARGS+=("--gpus" "all")
    IMAGE_NAME="${IMAGE_NAME:-lainbow-inference-api:latest}"
    ;;
  rocm)
    GPU_ARGS+=("--device" "/dev/kfd" "--device" "/dev/dri" "--group-add" "video" "--group-add" "render")
    IMAGE_NAME="${IMAGE_NAME:-lainbow-inference-api-rocm:latest}"
    ;;
  none)
    IMAGE_NAME="${IMAGE_NAME:-lainbow-inference-api:latest}"
    ;;
  *)
    echo "Unsupported INFERENCE_RUNTIME value: $RUNTIME (expected: cuda, rocm, none)" >&2
    exit 1
    ;;
esac

if [[ -z "$IMAGE_NAME" ]]; then
  echo "Unable to determine Docker image name. Set INFERENCE_DOCKER_IMAGE." >&2
  exit 1
fi

# shellcheck disable=SC2068
exec docker run --rm \
  "${GPU_ARGS[@]}" \
  --env-file "$PROJECT_ROOT/.env" \
  --network host \
  --add-host "host.docker.internal:host-gateway" \
  -v "$PROJECT_ROOT:/workspace" \
  -v "$AUDIO_DIR:$AUDIO_DIR:ro" \
  -w /workspace \
  "$IMAGE_NAME" \
  python /workspace/debug_inference.py "$MODEL_NAME" "$ABS_AUDIO_PATH" $@

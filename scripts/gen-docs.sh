#!/bin/sh
# Regenerates docs/configuration.md from the settings classes.
set -eu

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

uv run --no-sync python -m lainbow.core.settings_doc >docs/configuration.md

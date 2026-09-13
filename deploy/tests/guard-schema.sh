#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
require_operator
resolve_image
if [[ ${1:-} == --cpu ]]; then
  image_python /deploy/tests/guard_schema.py --guard-path /deploy/overrides/dsv41_guard.py
else
  own_container "$ENGINE_NAME" && engine_health || fail 'The owned model engine is not ready'
  docker exec "$ENGINE_NAME" /usr/bin/python3 /offline-tests/guard_schema.py --live --output /state/hf005-schema-live.json
fi

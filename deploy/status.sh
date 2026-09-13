#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
docker ps -a --filter label=dsv41.owner=offline-delivery --format '{{.Names}}  {{.Image}}  {{.Status}}'
if [[ -f "$STATE_DIR/capacity.json" ]]; then cat "$STATE_DIR/capacity.json"; fi
if own_container "$ENGINE_NAME"; then docker logs --tail 30 "$ENGINE_NAME" 2>&1; fi

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
require_operator
resolve_image
own_container "$ENGINE_NAME" && engine_health || fail 'Start the engine first'
mode=${1:-full-window}
case "$mode" in full-window|reuse|stability) ;; *) fail 'Usage: extended.sh full-window|reuse|stability [hours]' ;; esac
run="extended-$mode-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$STATE_DIR/$run"
docker exec -d "$ENGINE_NAME" /usr/bin/python3 /offline-tests/monitor.py \
  --output "/state/$run/resources.jsonl" --stop-file "/state/$run/monitor.stop"
trap 'touch "$STATE_DIR/$run/monitor.stop"' EXIT
docker exec "$ENGINE_NAME" /usr/bin/python3 /offline-tests/extended.py --mode "$mode" --hours "${2:-2}" \
  --base-url http://127.0.0.1:8000 --key-file /state/api-key.txt --capacity /state/capacity.json \
  --output "/state/$run/result.json"
docker logs --timestamps "$ENGINE_NAME" > "$STATE_DIR/$run/engine.log" 2>&1
printf 'Extended report: %s/%s/result.json\n' "$STATE_DIR" "$run"

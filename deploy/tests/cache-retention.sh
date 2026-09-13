#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
require_operator
resolve_image
exec 9>"$STATE_DIR/operation.lock"
flock -n 9 || fail 'Another deployment operation is active'
own_container "$ENGINE_NAME" && engine_health || fail 'The owned model engine is not ready'
action=${1:-}
if [[ "$action" == seed ]]; then
  run="cache-retention-$(date -u +%Y%m%dT%H%M%S)-$$"
elif [[ "$action" == check ]]; then
  [[ -f "$STATE_DIR/cache-retention-active.txt" ]] || fail 'Run tests/cache-retention.sh seed first'
  run=$(cat "$STATE_DIR/cache-retention-active.txt")
else
  fail 'Usage: tests/cache-retention.sh seed|check'
fi
[[ "$run" =~ ^cache-retention-[0-9]{8}T[0-9]{6}-[0-9]+$ ]] || fail 'Invalid saved probe path'
docker exec "$ENGINE_NAME" /usr/bin/python3 /offline-tests/cache_retention.py "$action" --directory "/state/$run"
if [[ "$action" == seed ]]; then
  printf '%s\n' "$run" > "$STATE_DIR/cache-retention-active.txt"
  printf '\nLeave the service idle. After the chosen interval, run: bash ./tests/cache-retention.sh check\n'
fi
printf 'Reports: %s/%s\n' "$STATE_DIR" "$run"

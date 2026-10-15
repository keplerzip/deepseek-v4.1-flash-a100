#!/usr/bin/env bash
set -euo pipefail
# --isolate-clients pauses only this delivery's local access frontend. The
# eight-GPU engine keeps running, and the EXIT trap restores the frontend.
# Samples remain non-signoff if direct internal clients still add traffic.
isolate_clients=false
benchmark_args=()
for argument in "$@"; do
  case "$argument" in
    --isolate-clients) isolate_clients=true ;;
    --help|-h)
      printf '%s\n' \
        'Usage: bash tests/performance-benchmark.sh [--isolate-clients] [--concurrency 64|32] [--images 1|5|8 | --request-file /state/private.json]' \
        'Default: C1 samples and C64 load, off/high, k=5, 1024 output tokens.' \
        '--isolate-clients stops only the owned local frontend and restores it on EXIT; it does not restart the engine or touch NewAPI.' \
        'Other direct internal clients can still contaminate results. CONTAMINATED reports retain actual client samples but fail performance signoff.' \
        'Private request files must already be available inside the engine, for example under /state.'
      exit 0 ;;
    *) benchmark_args+=("$argument") ;;
  esac
done
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
require_operator
exec 9>"$STATE_DIR/operation.lock"
flock -n 9 || fail 'Another deployment operation is active'
resolve_image
own_container "$ENGINE_NAME" && engine_health || fail 'Start the engine first'
restore_frontend=false
cleanup() {
  local result=$?
  trap - EXIT INT TERM
  if [[ "$restore_frontend" == true ]]; then
    printf 'Restoring this deployment frontend; the model was not restarted.\n'
    if ! bash "$DEPLOY_DIR/scripts/local-access.sh"; then
      printf 'ERROR: Frontend restoration failed; run bash scripts/local-access.sh. The engine is still running.\n' >&2
      ((result != 0)) || result=1
    fi
  fi
  exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$isolate_clients" == true ]]; then
  if docker inspect "$FRONTEND_NAME" >/dev/null 2>&1; then
    own_container "$FRONTEND_NAME" || fail 'Refusing to stop a frontend owned by another deployment'
    frontend_source=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/frontend.py"}}{{.Source}}{{end}}{{end}}' "$FRONTEND_NAME")
    access_source=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/access.json"}}{{.Source}}{{end}}{{end}}' "$FRONTEND_NAME")
    [[ "$frontend_source" == "$DEPLOY_DIR/scripts/local_access_frontend.py" && "$access_source" == "$STATE_DIR/local-access.json" ]] \
      || fail 'Refusing to stop a frontend mounted from another deployment directory'
    if [[ $(docker inspect --format '{{.State.Running}}' "$FRONTEND_NAME") == true ]]; then
      restore_frontend=true
      printf 'Pausing this deployment frontend for isolated internal benchmarking; engine remains running.\n'
      docker stop -t 90 "$FRONTEND_NAME" >/dev/null
    fi
  fi
  if [[ "$restore_frontend" == true ]]; then
    benchmark_args+=(--clients-isolated)
  else
    printf 'This deployment frontend was already unavailable; its original state will be retained.\n'
  fi
fi
run="performance-benchmark-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$STATE_DIR/$run"
current=$(cat "$STATE_DIR/current-run.txt")
[[ ! -f "$STATE_DIR/$current/performance.json" ]] || cp "$STATE_DIR/$current/performance.json" "$STATE_DIR/$run/performance.json"
cp "$STATE_DIR/runtime.resolved.json" "$STATE_DIR/$run/config.json"
cp "$STATE_DIR/capacity.json" "$STATE_DIR/$run/capacity.json"
failed=0
for mode in off high; do
  if ! docker exec "$ENGINE_NAME" /usr/bin/python3 /offline-tests/dspark_benchmark.py \
    --seed dsv41-performance-20260915 --mode "$mode" --k 5 \
    --output "/state/$run/$mode.json" "${benchmark_args[@]}" 2>&1 | tee "$STATE_DIR/$run/$mode.log"; then
    failed=1
  fi
done
printf 'C1/load k5 reports and DSpark acceptance counters: %s/%s\n' "$STATE_DIR" "$run"
exit "$failed"

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
require_operator
exec 9>"$STATE_DIR/operation.lock"
flock -n 9 || fail 'Another deployment operation is active'
resolve_image
[[ -f "$STATE_DIR/model-verification.json" ]] || fail 'Run deploy.sh to verify the complete model snapshot first'
[[ $(op get /state/model-verification.json status) == PASS ]] || fail 'Full model verification has not passed'
current_fingerprint=$(op fingerprint --model /models/DeepSeek-V4.1-Flash --manifest /deploy/manifests/model-files.json)
[[ -f "$STATE_DIR/model-stat-fingerprint.txt" && "$current_fingerprint" == "$(cat "$STATE_DIR/model-stat-fingerprint.txt")" ]] || fail 'Model files or location changed; run verify.sh again'
bash "$DEPLOY_DIR/scripts/preflight.sh"
if own_container "$ENGINE_NAME" && engine_health && own_container "$FRONTEND_NAME" && frontend_health; then
  [[ $(docker inspect --format '{{.Config.Image}}' "$ENGINE_NAME") == "$RUNTIME_IMAGE" ]] || fail 'A different runtime release is running; stop it before applying this one'
  printf 'Existing deployment is healthy. Use stop.sh before changing configuration.\n'
  exit 0
fi
remove_own "$FRONTEND_NAME"
remove_own "$ENGINE_NAME"
active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z "$active" ]] || fail 'Another process still occupies the GPUs'
{
  performance_signature="$(sha256sum "$DEPLOY_DIR/overrides/performance/manifest.json" | cut -d ' ' -f1)|$RUNTIME_IMAGE|$PERF_MHC$PERF_INDEXER$PERF_MOE_ALIGN|$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u | tr '\n' ',')"
  if [[ ! -f "$STATE_DIR/performance-kernels-approved.txt" || "$(cat "$STATE_DIR/performance-kernels-approved.txt")" != "$performance_signature" ]]; then
    printf 'Checking R1.2 SM80 kernels and EP8 communication once before model loading.\n'
    bash "$DEPLOY_DIR/tests/performance-kernels.sh"
    bash "$DEPLOY_DIR/tests/performance-ep8.sh"
    printf '%s\n' "$performance_signature" > "$STATE_DIR/performance-kernels-approved.txt.tmp"
    mv -- "$STATE_DIR/performance-kernels-approved.txt.tmp" "$STATE_DIR/performance-kernels-approved.txt"
  fi
}
make_secret
if docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
  [[ $(docker network inspect --format '{{.Internal}} {{index .Labels "dsv41.owner"}}' "$NETWORK_NAME") == 'true offline-delivery' ]] || fail 'Existing network has a different owner or permits external routing'
else
  docker network create --internal --label dsv41.owner=offline-delivery "$NETWORK_NAME" >/dev/null
fi
concurrency=32
if [[ -f "$STATE_DIR/capacity.json" ]] && [[ $(op get /state/capacity.json topology 2>/dev/null || true) == TP4-DP2-EP8 ]]; then
  concurrency=$(op get /state/capacity.json required_concurrency)
fi
config=/deploy/configs/runtime.json
[[ ! -f "$STATE_DIR/selected-config.json" ]] || config=/state/selected-config.json
[[ ! -f "$STATE_DIR/tuning-active.json" ]] || config=/state/tuning-active.json
mode="$NETWORK_NAME"
[[ ${1:-} != --offline-first ]] || mode=none
completed=false
seen_concurrency=" $concurrency "
minimum_blocks=0
fixed_blocks=()
trap 'docker logs --timestamps "$ENGINE_NAME" > "$STATE_DIR/last-engine.log" 2>&1 || true' EXIT
for attempt in 1 2 3 4 5 6; do
  run="run-$(date -u +%Y%m%dT%H%M%S)-$attempt"
  mkdir -p "$STATE_DIR/$run/observations"
  op resolve --config "$config" --concurrency "$concurrency" --output "/state/$run/resolved.json" "${fixed_blocks[@]}"
  launch_engine "$mode" "$run"
  wait_engine
  op capacity --directory "/state/$run/observations" --concurrency "$concurrency" --output "/state/$run/capacity.json"
  required=$(op get "/state/$run/capacity.json" required_concurrency)
  pool=$(op get "/state/$run/capacity.json" minimum_pool_blocks)
  if ((minimum_blocks==0 || pool<minimum_blocks)); then minimum_blocks=$pool; fi
  docker logs --timestamps "$ENGINE_NAME" > "$STATE_DIR/$run/engine.log" 2>&1
  if [[ "$required" != "$concurrency" ]]; then
    if [[ "$seen_concurrency" == *" $required "* ]]; then
      fixed_blocks=(--fixed-blocks "$minimum_blocks")
      printf 'Capacity iteration repeated C=%s; fixing KV to the smallest actually profiled pool (%s blocks) and revalidating.\n' "$required" "$minimum_blocks"
    fi
    printf 'KV calibration requests C=%s (current C=%s); restarting with the exact formula.\n' "$required" "$concurrency"
    concurrency=$required
    seen_concurrency+="$concurrency "
    remove_own "$ENGINE_NAME"
    continue
  fi
  op graphs --resolved "/state/$run/resolved.json" --directory "/state/$run/observations" --concurrency "$concurrency" --output "/state/$run/graphs.json"
  for dp_rank in 0 1; do
    docker exec "$ENGINE_NAME" /usr/bin/python3 /offline-tests/acceptance.py --suite smoke --dp-rank "$dp_rank" \
      --base-url http://127.0.0.1:8000 --key-file /state/api-key.txt --output "/state/$run/smoke-dp$dp_rank.json"
  done
  op replays --resolved "/state/$run/resolved.json" --directory "/state/$run/observations" --concurrency "$concurrency" --output "/state/$run/replays.json"
  if [[ "$mode" == none ]]; then
    if [[ ${2:-} == --full-offline-proof ]]; then
      docker exec "$ENGINE_NAME" /usr/bin/python3 /offline-tests/acceptance.py --suite full \
        --base-url http://127.0.0.1:8000 --key-file /state/api-key.txt \
        --concurrency "$concurrency" --output "/state/$run/offline-full.json"
    fi
    docker exec "$ENGINE_NAME" /usr/bin/python3 -c \
      'import pathlib,json; r=pathlib.Path("/proc/net/route").read_text(); assert not any(x.split()[1]=="00000000" for x in r.splitlines()[1:]); print(json.dumps({"status":"PASS","network":"none","routes":r}))' \
      > "$STATE_DIR/$run/offline-network.json"
    printf 'First-start smoke passed in network=none; verifying a service restart on the internal network.\n'
    mode="$NETWORK_NAME"
    remove_own "$ENGINE_NAME"
    continue
  fi
  cp "$STATE_DIR/$run/capacity.json" "$STATE_DIR/capacity.json"
  cp "$STATE_DIR/$run/resolved.json" "$STATE_DIR/runtime.resolved.json"
  completed=true
  break
done
[[ "$completed" == true ]] || fail 'Capacity did not converge within six launches; reports retained, no context or concurrency downgrade applied'
bash "$DEPLOY_DIR/scripts/local-access.sh"
printf 'Model: DeepSeek-V4.1-Flash\nContext: 262144\nConcurrency: %s\nAPI port: %s\nClient API key: not required (loopback and Docker bridges only)\n' "$concurrency" "$API_PORT"
printf 'GPU smoke/capacity passed. Full acceptance, speed selection and stability require tests/run-all.sh.\n'

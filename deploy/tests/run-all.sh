#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
require_operator
resolve_image
own_container "$ENGINE_NAME" && engine_health || fail 'Start the engine first'
concurrency=$(op get /state/capacity.json required_concurrency)
run="acceptance-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$STATE_DIR/$run"
docker exec -d "$ENGINE_NAME" /usr/bin/python3 /offline-tests/monitor.py \
  --output "/state/$run/resources.jsonl" --stop-file "/state/$run/monitor.stop"
trap 'touch "$STATE_DIR/$run/monitor.stop"' EXIT
base=http://127.0.0.1:8000
for suite in full long; do
  for dp_rank in 0 1; do
    docker exec "$ENGINE_NAME" /usr/bin/python3 /offline-tests/acceptance.py \
      --suite "$suite" --base-url "$base" --key-file /state/api-key.txt --dp-rank "$dp_rank" \
      --concurrency "$((concurrency/2))" --output "/state/$run/$suite-dp$dp_rank.json"
  done
done
docker exec "$ENGINE_NAME" /usr/bin/python3 /offline-tests/acceptance.py \
  --suite parallel --base-url "$base" --key-file /state/api-key.txt \
  --concurrency "$concurrency" --output "/state/$run/global-concurrency.json"
if ((concurrency>32)); then
  docker exec "$ENGINE_NAME" /usr/bin/python3 /offline-tests/acceptance.py \
    --suite parallel --base-url "$base" --key-file /state/api-key.txt \
    --concurrency 32 --output "/state/$run/c32.json"
fi
docker exec "$ENGINE_NAME" /usr/bin/python3 /offline-tests/benchmark.py --base-url "$base" \
  --key-file /state/api-key.txt --concurrency "$concurrency" --requests 200 --thinking \
  --output "/state/$run/benchmark-thinking.json"
docker logs --timestamps "$ENGINE_NAME" > "$STATE_DIR/$run/engine.log" 2>&1
printf 'Functional and benchmark reports: %s/%s\n' "$STATE_DIR" "$run"
printf 'Extended stability and offline cache-rebuild gates remain separate. See tests/README.zh-CN.md.\n'

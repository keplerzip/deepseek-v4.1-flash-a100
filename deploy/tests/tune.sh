#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
require_operator
resolve_image
run="tuning-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$STATE_DIR/$run"
printf '[]\n' > "$STATE_DIR/$run/matrix.json"
trap 'rm -f -- "$STATE_DIR/tuning-active.json"' EXIT
# Interleave memory/batch choices; every launch calibrates C against its own KV.
for candidate in 0.90:4096 0.90:8192 0.90:16384 0.92:8192 0.94:8192; do
  mem=${candidate%:*}; batch=${candidate#*:}; name="m${mem}-b${batch}"
  mkdir -p "$STATE_DIR/$run/$name"
  bash "$DEPLOY_DIR/stop.sh"
  image_python -c 'import json,sys,pathlib; p=json.loads(pathlib.Path("/deploy/configs/runtime.json").read_text());p.update(gpu_memory_utilization=float(sys.argv[1]),max_num_batched_tokens=int(sys.argv[2]));pathlib.Path("/state/tuning-active.json").write_text(json.dumps(p,indent=2)+"\n")' "$mem" "$batch"
  cp "$STATE_DIR/tuning-active.json" "$STATE_DIR/$run/$name/config.json"
  state=FAIL
  if bash "$DEPLOY_DIR/start.sh" > "$STATE_DIR/$run/$name/start.log" 2>&1; then
    if docker exec "$ENGINE_NAME" /usr/bin/python3 /offline-tests/acceptance.py --suite full \
      --base-url http://127.0.0.1:8000 --key-file /state/api-key.txt --concurrency 32 \
      --output "/state/$run/$name/full.json" > "$STATE_DIR/$run/$name/full.log" 2>&1 && \
      docker exec "$ENGINE_NAME" /usr/bin/python3 /offline-tests/benchmark.py --thinking \
      --base-url http://127.0.0.1:8000 --key-file /state/api-key.txt --concurrency 32 --requests 200 \
      --output "/state/$run/$name/benchmark.json" > "$STATE_DIR/$run/$name/benchmark.log" 2>&1; then
      state=PASS
      cp "$STATE_DIR/capacity.json" "$STATE_DIR/$run/$name/capacity.json"
    fi
  fi
  image_python -c 'import json,sys,pathlib;root=pathlib.Path(sys.argv[1]); name=sys.argv[2]; d=root/name; rows=json.loads((root/"matrix.json").read_text());r={"name":name,"status":sys.argv[3],"config":json.loads((d/"config.json").read_text())}; p=d/"benchmark.json";r["benchmark"]=json.loads(p.read_text()) if p.exists() else {};rows.append(r);(root/"matrix.json").write_text(json.dumps(rows,indent=2)+"\n")' "/state/$run" "$name" "$state"
  [[ "$candidate" != 0.90:4096 || "$state" == PASS ]] || fail 'Baseline failed; fix its report before tuning'
done
image_python /deploy/tests/select_tuning.py "/state/$run"
bash "$DEPLOY_DIR/stop.sh"
cp "$STATE_DIR/$run/selected-config.json" "$STATE_DIR/selected-config.json"
rm -f -- "$STATE_DIR/tuning-active.json"
bash "$DEPLOY_DIR/start.sh"
printf 'One candidate selected. Complete run-all.sh, extended full-window/stability and offline-rebuild before performance sign-off. Report: %s/%s/selection.json\n' "$STATE_DIR" "$run"

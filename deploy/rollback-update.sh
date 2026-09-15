#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
require_operator
resolve_image
[[ -f "$STATE_DIR/latest-update.json" ]] || fail 'No source-update receipt exists'
image_python -c 'import json,pathlib; assert json.loads(pathlib.Path("/state/latest-update.json").read_text())["status"] == "APPLIED", "Latest update is not applied"'
bash "$DEPLOY_DIR/stop.sh"
if ! docker run --rm --pull never --network none --user "$(id -u):$(id -g)" \
  --entrypoint /usr/bin/python3 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$DEPLOY_DIR:/target" "$RUNTIME_IMAGE" \
  /target/scripts/update_manager.py --target /target --rollback; then
  bash "$DEPLOY_DIR/start.sh" || true
  fail 'Rollback refused; inspect the preserved receipt and changed files'
fi
exec bash "$DEPLOY_DIR/start.sh"

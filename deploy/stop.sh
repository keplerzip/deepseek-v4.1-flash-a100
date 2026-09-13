#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
require_operator
exec 9>"$STATE_DIR/operation.lock"
flock -n 9 || fail 'Another deployment operation is active'
remove_own "$FRONTEND_NAME"
if own_container "$ENGINE_NAME"; then docker logs --timestamps "$ENGINE_NAME" > "$STATE_DIR/last-engine.log" 2>&1; fi
remove_own "$ENGINE_NAME"
printf 'Stopped this deployment. Weights, validation reports and local compilation caches retained.\n'

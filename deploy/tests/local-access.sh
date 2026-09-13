#!/usr/bin/env bash
# Keyless access from the host and Docker; does not change existing containers.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
require_operator
resolve_image
own_container "$ENGINE_NAME" && engine_health && frontend_health || fail 'Start the local-only service first'
run="local-access-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$STATE_DIR/$run"
for mode in host bridge; do
  base="http://127.0.0.1:$API_PORT"
  args=(--network host)
  if [[ "$mode" == bridge ]]; then
    base="http://host.docker.internal:$API_PORT"
    args=(--network bridge --add-host host.docker.internal:host-gateway)
  fi
  docker run --rm --pull never "${args[@]}" --read-only --cap-drop ALL \
    --security-opt no-new-privileges "${CONTAINER_USER_ARGS[@]}" \
    --mount "type=bind,src=$DEPLOY_DIR/scripts/local_access_frontend.py,dst=/check.py,readonly" \
    -v "$STATE_DIR:/state" --entrypoint /usr/bin/python3 "$RUNTIME_IMAGE" \
    /check.py check --base-url "$base" --output "/state/$run/$mode.json"
done
printf 'Anonymous host and Docker access reports: %s/%s\n' "$STATE_DIR" "$run"

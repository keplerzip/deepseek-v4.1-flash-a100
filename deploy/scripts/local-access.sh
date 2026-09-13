#!/usr/bin/env bash
# Called by start.sh while it holds the deployment operation lock.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_operator
resolve_image
own_container "$ENGINE_NAME" && engine_health || fail 'The owned engine is not ready'
upstream=$(docker inspect --format "{{(index .NetworkSettings.Networks \"$NETWORK_NAME\").IPAddress}}" "$ENGINE_NAME")
mapfile -t dsv41_bridges < <(docker network ls --filter driver=bridge --format '{{.ID}}')
((${#dsv41_bridges[@]})) || fail 'No local Docker bridge networks found'
docker network inspect "${dsv41_bridges[@]}" > "$STATE_DIR/local-access-networks.json"
image_python /deploy/scripts/local_access_frontend.py configure \
  --networks /state/local-access-networks.json --upstream "$upstream" --port "$API_PORT" \
  --output /state/local-access.json
remove_own "$FRONTEND_NAME"
docker run -d --pull never --name "$FRONTEND_NAME" --label dsv41.owner=offline-delivery \
  --network host --read-only --cap-drop ALL --security-opt no-new-privileges --user 65534:65534 \
  --mount "type=bind,src=$DEPLOY_DIR/scripts/local_access_frontend.py,dst=/frontend.py,readonly" \
  --mount "type=bind,src=$STATE_DIR/local-access.json,dst=/access.json,readonly" \
  --entrypoint /usr/bin/python3 "$RUNTIME_IMAGE" /frontend.py serve --config /access.json >/dev/null
for dsv41_try in {1..30}; do
  if frontend_health; then
    docker exec "$FRONTEND_NAME" /usr/bin/python3 -c \
      'import json,sys,urllib.request;op=urllib.request.build_opener(urllib.request.ProxyHandler({}));data=json.load(op.open("http://127.0.0.1:"+sys.argv[1]+"/v1/models",timeout=10));ids=[m["id"] for m in data["data"]];assert ids==["DeepSeek-V4.1-Flash"],ids;print(json.dumps({"status":"PASS","client_api_key_required":False,"models":ids}))' "$API_PORT"
    exit 0
  fi
  [[ $(docker inspect --format '{{.State.Running}}' "$FRONTEND_NAME") == true ]] || break
  sleep 1
done
docker logs --tail 30 "$FRONTEND_NAME" >&2 || true
fail 'Local-only frontend failed its anonymous health/models check'

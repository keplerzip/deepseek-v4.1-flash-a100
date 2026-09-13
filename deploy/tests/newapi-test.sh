#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
require_operator
resolve_image
: "${DSV41_NEWAPI_TOKEN_FILE:?Set DSV41_NEWAPI_TOKEN_FILE to a file containing the NewAPI model-limited token}"
[[ -n "$NEWAPI_BASE_URL" ]] || fail 'Set NEWAPI_BASE_URL in deployment.env to the NewAPI root URL'
[[ -f "$DSV41_NEWAPI_TOKEN_FILE" ]] || fail 'NewAPI token file is missing'
run="newapi-$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$STATE_DIR/$run"
docker run --rm --pull never "${CONTAINER_USER_ARGS[@]}" --network host --read-only --tmpfs /tmp:rw,size=512m \
  --entrypoint /usr/bin/python3 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$DEPLOY_DIR/tests:/offline-tests:ro" -v "$STATE_DIR:/state" \
  -v "$(realpath -e -- "$DSV41_NEWAPI_TOKEN_FILE"):/newapi-token:ro" \
  "$RUNTIME_IMAGE" /offline-tests/acceptance.py --suite gateway \
  --base-url "$NEWAPI_BASE_URL" --key-file /newapi-token \
  --concurrency 32 --output "/state/$run/gateway.json"
printf 'Match the cache test request ID and cached_tokens with NewAPI usage logs (Cache↓): %s/%s/gateway.json\n' "$STATE_DIR" "$run"

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/common.sh"
require_operator
resolve_image
# This explicit maintenance entry restarts only containers owned by this release.
bash "$DEPLOY_DIR/stop.sh"
run="cache-before-rebuild-$(date -u +%Y%m%dT%H%M%S)"
if [[ -d "$STATE_DIR/cache" ]]; then mv -- "$STATE_DIR/cache" "$STATE_DIR/$run"; fi
bash "$DEPLOY_DIR/start.sh" --offline-first --full-offline-proof
printf 'Cache rebuilt from packaged inputs. Previous derived cache retained at %s/%s.\n' "$STATE_DIR" "$run"

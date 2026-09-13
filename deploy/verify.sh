#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$DEPLOY_DIR"
sha256sum --quiet -c manifests/deploy.sha256
source "$DEPLOY_DIR/scripts/common.sh"
resolve_image
before=$(op fingerprint --model /models/DeepSeek-V4.1-Flash --manifest /deploy/manifests/model-files.json)
image_python /opt/dsv41-offline/verify_model.py \
  --model /models/DeepSeek-V4.1-Flash --manifest /deploy/manifests/model-files.json \
  --full-hash --output /state/model-verification.json
after=$(op fingerprint --model /models/DeepSeek-V4.1-Flash --manifest /deploy/manifests/model-files.json)
[[ "$before" == "$after" ]] || fail 'Model files changed during full verification'
printf '%s\n' "$after" > "$STATE_DIR/model-stat-fingerprint.txt"
printf 'Deployment components and full model snapshot verified.\n'

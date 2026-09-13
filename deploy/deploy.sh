#!/usr/bin/env bash
set -euo pipefail
DEPLOY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$DEPLOY_DIR"
sha256sum --quiet -c manifests/deploy.sha256
source "$DEPLOY_DIR/scripts/common.sh"
require_operator
bash "$DEPLOY_DIR/scripts/preflight.sh"
if ! docker image inspect deepseek-v4.1-flash-a100:20260913-r1 >/dev/null 2>&1; then
  docker load -i images/runtime-image.tar
fi
bash "$DEPLOY_DIR/verify.sh"
if [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) ]]; then
  bash "$DEPLOY_DIR/tests/gpu-probe.sh"
fi
bash "$DEPLOY_DIR/start.sh" --offline-first

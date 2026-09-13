#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/common.sh"
require_operator
resolve_image
output="dsv41-results-$(date -u +%Y%m%dT%H%M%S).tar.gz"
image_python /deploy/scripts/export_results.py "/state/$output"
printf 'Review and copy this result archive back: %s/%s\n' "$STATE_DIR" "$output"

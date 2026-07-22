#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "scripts/ci_lite_all.sh is deprecated; use scripts/ci_headers_all.sh." >&2
exec "${ROOT_DIR}/scripts/ci_headers_all.sh" "$@"

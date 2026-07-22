#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "scripts/ci_full_all.sh is deprecated; use scripts/ci_native_all.sh instead." >&2
exec "${ROOT_DIR}/scripts/ci_native_all.sh" "$@"

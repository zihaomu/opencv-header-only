#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEADER_ROOT="${ROOT_DIR}/include"

FORBIDDEN_INCLUDE_REGEX='#include[[:space:]]*[<"](test/|src/|backend/|libnpy/)'

if command -v rg >/dev/null 2>&1; then
  if rg -n "${FORBIDDEN_INCLUDE_REGEX}" "${HEADER_ROOT}" -g '!include/cvh/3rdparty/**'; then
    echo "Found forbidden include path in public headers." >&2
    exit 1
  fi
else
  if grep -RInE "${FORBIDDEN_INCLUDE_REGEX}" "${HEADER_ROOT}" --exclude-dir=3rdparty; then
    echo "Found forbidden include path in public headers." >&2
    exit 1
  fi
fi

echo "Public header dependency check passed."

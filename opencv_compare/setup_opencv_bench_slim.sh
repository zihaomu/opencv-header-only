#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPARE_DIR="${ROOT_DIR}/opencv_compare"

OPENCV_REPO="${CVH_OPENCV_REPO:-https://github.com/zihaomu/opencv.git}"
OPENCV_BRANCH="${CVH_OPENCV_BRANCH:-opencv-bench-slim-v4.13}"
OPENCV_DIR="${CVH_OPENCV_DIR:-${COMPARE_DIR}/opencv-bench-slim}"
OPENCV_BUILD_DIR="${CVH_OPENCV_BUILD_DIR:-${OPENCV_DIR}/build-slim}"

DO_BUILD=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--build] [--help]

Options:
  --build    Run opencv-bench-slim build script after clone/update.
  --help     Show this help message.

Environment:
  CVH_OPENCV_REPO       (default: ${OPENCV_REPO})
  CVH_OPENCV_BRANCH     (default: ${OPENCV_BRANCH})
  CVH_OPENCV_DIR        (default: ${OPENCV_DIR})
  CVH_OPENCV_BUILD_DIR  (default: ${OPENCV_BUILD_DIR})
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      DO_BUILD=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$(dirname "${OPENCV_DIR}")"

if [[ ! -d "${OPENCV_DIR}/.git" ]]; then
  if [[ -d "${OPENCV_DIR}" ]] && [[ -n "$(ls -A "${OPENCV_DIR}" 2>/dev/null || true)" ]]; then
    echo "Existing non-git directory blocks clone: ${OPENCV_DIR}" >&2
    exit 2
  fi

  git clone --depth=1 --branch "${OPENCV_BRANCH}" "${OPENCV_REPO}" "${OPENCV_DIR}"
else
  git -C "${OPENCV_DIR}" remote set-url origin "${OPENCV_REPO}"
  git -C "${OPENCV_DIR}" fetch --depth=1 origin "${OPENCV_BRANCH}"
  git -C "${OPENCV_DIR}" checkout -B "${OPENCV_BRANCH}" FETCH_HEAD
fi

if [[ ${DO_BUILD} -eq 1 ]]; then
  if [[ ! -f "${OPENCV_DIR}/build_opencv.bash" ]]; then
    echo "Missing build_opencv.bash in ${OPENCV_DIR}" >&2
    exit 2
  fi

  mkdir -p "${OPENCV_BUILD_DIR}"
  (
    cd "${OPENCV_BUILD_DIR}"
    bash "../build_opencv.bash"
  )
fi

echo "opencv_bench_slim_ready: repo=${OPENCV_REPO}, branch=${OPENCV_BRANCH}, dir=${OPENCV_DIR}, build=${DO_BUILD}"

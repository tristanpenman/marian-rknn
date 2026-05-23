#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "docker" ]]; then
  shift
  cd "${ROOT_DIR}"
  exec docker compose run --rm --remove-orphans --build native ./scripts/build-native.sh "$@"
fi

BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-native}"
BUILD_TYPE="${1:-Release}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_CXX_COMPILER="${CXX:-aarch64-linux-gnu-g++}" \
  -DBUILD_TESTING=OFF

make -C "${BUILD_DIR}" -j"$(nproc)"

echo "Native Rockchip Linux build finished: ${BUILD_DIR}/marian-rknn"

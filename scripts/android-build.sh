#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NDK_PATH="${ANDROID_NDK_HOME:-}"
RKNN_RUNTIME_SO="${ROOT_DIR}/thirdparty/rknpu2/lib-android/arm64-v8a/librknnrt.so"
BUILD_TYPE="Release"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --android-ndk-home)
      NDK_PATH="$2"
      shift 2
      ;;
    --android-ndk-home=*)
      NDK_PATH="${1#*=}"
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      BUILD_TYPE="$1"
      shift
      ;;
  esac
done

if [[ -z "${NDK_PATH}" ]]; then
  cat <<USAGE
Usage: $0 [--android-ndk-home <ndk-path>] [build-type]

NDK path is resolved in order of precedence:
  1. --android-ndk-home option
  2. ANDROID_NDK_HOME environment variable

Example:
  $0 --android-ndk-home /opt/android-ndk-r26d Release
USAGE
  exit 1
fi

BUILD_DIR="${ROOT_DIR}/build-android"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-34 \
  -DCMAKE_TOOLCHAIN_FILE="${NDK_PATH}/build/cmake/android.toolchain.cmake" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DBUILD_TESTING=OFF \
  -DMARIAN_RKNN_BUILD_BENCHMARK=ON \
  -DRKNN_RUNTIME_LIB="${RKNN_RUNTIME_SO}"

cmake --build "${BUILD_DIR}" -- -j"$(nproc)"

echo "Android build finished: ${BUILD_DIR}/marian-rknn"

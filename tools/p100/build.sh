#!/usr/bin/env bash
# Linux build for the Tesla P100 (sm_60) product target. Mirrors tools/v100/build.sh with the
# Pascal toolchain window: CUDA 12.x only (13 removed Pascal compilation), pinned here to the
# 12.4 floor that the root CMakeLists enforces for arch 60. Dependencies (FFmpeg/CURL) reuse
# the shared build_dependencies.sh; NVLink/PCIe P2P probing lives in tools/tp2 (see its sm_60
# build lines) and docs/p100-migration.md section 4.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/../.." && pwd)

readonly deps_dir="${repo_dir}/build/_deps"
readonly build_dir="${repo_dir}/build-p100-duo"

NINFER_DEPS_DIR="${deps_dir}" "${script_dir}/../v100/build_dependencies.sh"

PKG_CONFIG_PATH="${deps_dir}/install/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}" \
    cmake -S "${repo_dir}" -B "${build_dir}" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=60 \
        -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build "${build_dir}" --target ninfer ninfer-serve -j

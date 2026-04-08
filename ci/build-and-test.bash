#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
KERNELS=(memcpy stencil matmul sigmoid)
TIMEOUT_PER_STATE=3  # seconds per nvbench state

# ---------------------------------------------------------------------------
# Configure & build
# ---------------------------------------------------------------------------
echo "=== Configuring ==="
cmake -B "${BUILD_DIR}" -S "${REPO_ROOT}" -DCMAKE_BUILD_TYPE=Release

TARGETS=()
for k in "${KERNELS[@]}"; do
  TARGETS+=(--target "bench_${k}")
done

echo "=== Building ${KERNELS[*]} ==="
cmake --build "${BUILD_DIR}" --parallel "${TARGETS[@]}"

# ---------------------------------------------------------------------------
# Run each benchmark
# ---------------------------------------------------------------------------
FAILED=()
for k in "${KERNELS[@]}"; do
  BIN="${BUILD_DIR}/bench_${k}"
  echo ""
  echo "=== Running bench_${k} ==="
  if "${BIN}" --timeout "${TIMEOUT_PER_STATE}"; then
    echo "--- bench_${k}: PASSED ---"
  else
    echo "--- bench_${k}: FAILED ---"
    FAILED+=("${k}")
  fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "==============================="
if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo "All ${#KERNELS[@]} benchmarks passed."
  exit 0
else
  echo "FAILED: ${FAILED[*]}"
  exit 1
fi

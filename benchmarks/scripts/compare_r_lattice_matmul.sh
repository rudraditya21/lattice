#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
SIZE="${1:-1024}"
ITERS="${2:-3}"
WARMUP="${3:-1}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --target lattice_bench

RESULTS_DIR="${ROOT_DIR}/benchmarks/results"
mkdir -p "${RESULTS_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

R_OUT="${RESULTS_DIR}/r_matmul_${SIZE}_${STAMP}.txt"
CPU_OUT="${RESULTS_DIR}/lattice_cpu_matmul_${SIZE}_${STAMP}.txt"
METAL_OUT="${RESULTS_DIR}/lattice_metal_matmul_${SIZE}_${STAMP}.txt"

if command -v Rscript >/dev/null 2>&1; then
  Rscript "${SCRIPT_DIR}/run_r_matmul.R" "${SIZE}" "${ITERS}" "${WARMUP}" | tee "${R_OUT}"
else
  echo "Rscript not found; skipping R baseline" | tee "${R_OUT}"
fi

"${BUILD_DIR}/lattice_bench" --backend cpu --ops matmul --matmul-size "${SIZE}" \
  --warmup "${WARMUP}" --iters "${ITERS}" | tee "${CPU_OUT}"

if [[ "$(uname -s)" == "Darwin" ]]; then
  "${BUILD_DIR}/lattice_bench" --backend metal --ops matmul --matmul-size "${SIZE}" \
    --warmup "${WARMUP}" --iters "${ITERS}" | tee "${METAL_OUT}"
else
  echo "Metal backend not available on this host" | tee "${METAL_OUT}"
fi

echo "Results saved in ${RESULTS_DIR}"

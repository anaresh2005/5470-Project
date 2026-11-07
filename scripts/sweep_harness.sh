#!/bin/bash
# Run inside a one-node/four-GPU allocation. All outputs are retained in results/.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-results/gpu-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"
{
  date -u
  git rev-parse HEAD
  git status --short
  nvidia-smi
  nvcc --version
  env | sort | grep -E '^(NCCL_|CUDA_VISIBLE_DEVICES|SLURM_JOB_ID|SLURM_JOB_NODELIST)=' || true
} > "$OUT/metadata.txt" 2>&1
for bytes in ${SIZES:-12288 12582912 50331648}; do
  for delay in ${DELAYS:-0 1 5}; do
    for gpu in ${STRAGGLERS:-3 0}; do
      log="$OUT/b${bytes}-d${delay}-g${gpu}.log"
      srun --ntasks=1 build/harness4 "$bytes" all "${ITERS:-10}" "$delay" "${WARMUPS:-5}" "$gpu" "${SEED:-5470}" | tee "$log"
    done
  done
done
python3 harness/report.py "$OUT"/*.log --out "$OUT/summary"
echo "Results saved to $OUT"

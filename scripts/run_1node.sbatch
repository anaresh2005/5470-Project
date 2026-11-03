#!/bin/bash
# Run a 4-GPU AllReduce on one node. Submit from the repo root:
#   sbatch scripts/run_1node.sbatch [bytes] [algorithm] [iters] [delay_ms]
# algorithm: stragglar | ring | rhd | direct
# delay_ms: straggler delay; -1 skips the pre-work phase
# Keep bytes divisible by 48 so chunks split evenly for every algorithm.
# Use a different binary with: sbatch --export=ALL,BIN=path/to/binary ...
#SBATCH -A m4999_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --gpus-per-node=4
#SBATCH -t 00:10:00
#SBATCH -J allreduce4
#SBATCH -o logs/%x-%j.out

set -euo pipefail

BYTES=${1:-12582912}
ALG=${2:-stragglar}
ITERS=${3:-10}
DELAY=${4:--1}
BIN=${BIN:-build/allreduce4}

source scripts/env.sh

echo "commit: $(git rev-parse --short HEAD)"
echo "job: $SLURM_JOB_ID on $SLURM_JOB_NODELIST"
echo "run: $BIN $BYTES $ALG $ITERS $DELAY"
nvidia-smi -L

# The program exits 0 even when values are wrong, so check its output here.
# Wrong values are capped at 20 lines to keep the log small.
srun "$BIN" "$BYTES" "$ALG" "$ITERS" "$DELAY" | awk '
  /^Rank [0-9]+, idx / { bad++; if (bad <= 20) print; next }
  /^Rank [0-9]+, done/ { done++ }
  { print }
  END {
    if (bad)       { printf "FAIL: %d wrong values\n", bad; exit 1 }
    if (done != 4) { printf "FAIL: only %d of 4 ranks finished\n", done; exit 1 }
    print "PASS: all ranks finished with correct values"
  }'

#!/bin/bash
#SBATCH -A m4999_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH --gpus-per-node=4
#SBATCH -t 00:10:00
#SBATCH -J harness4
#SBATCH -o logs/%x-%j.out
set -euo pipefail
source scripts/env.sh
make harness
bash scripts/sweep_harness.sh

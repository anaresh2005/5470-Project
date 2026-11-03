#!/bin/bash
# Multi-process send/recv test with one process per GPU. Submit from the repo root:
#   sbatch scripts/run_mp.sbatch [ints_per_buffer]
#SBATCH -A m4999_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-node=4
#SBATCH -t 00:10:00
#SBATCH -J mp_sendrecv
#SBATCH -o logs/%x-%j.out

set -euo pipefail

COUNT=${1:-1048576}

source scripts/env.sh

# MPI only carries the NCCL ID from host memory, so it doesn't need GPU support.
export MPICH_GPU_SUPPORT_ENABLED=0
# Log how NCCL connects the GPUs, to confirm which transport it uses.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,P2P,NET

echo "commit: $(git rev-parse --short HEAD)"
echo "job: $SLURM_JOB_ID on $SLURM_JOB_NODELIST"

# --gpu-bind=none lets every process see all the node's GPUs, so NCCL can use NVLink between them.
srun --gpu-bind=none --cpu-bind=cores build/mp_sendrecv "$COUNT"

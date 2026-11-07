# Midpoint results — Nov 7

Both Perlmutter jobs passed: smoke test `60968895` and sweep `60968954`.
The tested CUDA code is commit `9bd1b8e`; later changes only affect reporting
and documentation.

The sweep ran on four A100-SXM4-40GB GPUs with CUDA 13.2 and NCCL 2.29.2:
12 KiB/12 MiB/48 MiB per GPU, delays 0/1/5 ms, slow GPUs 3/0, seed 5470,
five warmups and ten samples per algorithm/case.

All **540 timed runs** matched NCCL on every GPU, with **zero maximum absolute
error**. NCCL also matched the CPU sum. Generated-code speedup ranged from
**0.315x to 0.994x** across 18 cases, so it was slower than NCCL in this sweep.
The four-GPU correctness milestone is done; performance tuning and the final
eight-GPU/two-node test remain.

Evidence:
- `job-logs/` and `job-accounting.txt`: complete logs and successful exit codes.
- `build.txt`: CUDA build command.
- `gpu-20261107T173045Z/`: raw case logs and machine metadata.
- `gpu-20261107T173045Z/summary/`: CSVs, plots, and results panel (PDF/PNG/text).
- `validation/`: host tests and n=4/8/16 schedule simulation.

Timing includes delay, ReduceScatter, communication and all-stream completion;
setup and validation are excluded. Speedups pair matching iterations within
each case. See [the harness notes](../harness/README.md) for commands and limits.

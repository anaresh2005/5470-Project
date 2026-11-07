# Aneesh's n=4 harness

This implements the midpoint responsibilities in `PLAN.md`. One process drives
four GPUs; the generated entry point is unchanged. The upstream handwritten
source and David's generator are not edited.

## Build and run on Perlmutter

```bash
source scripts/env.sh
make harness
# Uses Nathnael's existing launcher; defaults to five warmups and slow GPU 3.
sbatch --export=ALL,BIN=build/harness4 scripts/run_1node.sbatch 12582912 all 10 1
# Full sweep, including physical GPU 0 as the straggler:
sbatch scripts/run_harness.sbatch
```

Run inside an allocation, never on the login node:

```bash
build/harness4 12582912 all 10 1 5 3 5470
```

Arguments: `bytes algorithm iterations delay_ms [warmups=5]
[physical_straggler=3] [seed=5470]`. Algorithms: `nccl`, `handwritten`,
`generated`, `all`; `stragglar` is an alias for `handwritten` for launcher
compatibility. Bytes must be a positive multiple of 12 (48 also works).
`-1` is accepted as zero delay for the existing launcher, but the ReduceScatter
phase is **always** performed. Zero warmups is supported for smoke testing;
use at least five for reported performance.

The sweep defaults to 12 KiB, 12 MiB, and 48 MiB per GPU, delays 0/1/5 ms,
physical stragglers 3/0, five warmups and ten measured samples per algorithm.
Override `SIZES`, `DELAYS`, `STRAGGLERS`, `ITERS`, `WARMUPS`, `SEED`, or `OUT`
through the job environment. Job walltime bounds deadlock experiments.
Each completed case is kept even if a later case fails. Report completed logs
explicitly; never report a truncated log as success.

## Correctness and timing

- Deterministic signed inputs vary with physical GPU, element index, and seed.
  Their small binary fractions sum exactly in float32 for four GPUs. This catches
  offset/copy errors that uniform rank constants can hide. Use several seeds for
  additional coverage; arbitrary real-valued reduction accuracy is not established.
- NCCL AllReduce first produces a reference. Every element on every rank is
  checked against an independent CPU sum.
- Inputs and scratch are reset before **every** invocation. Every warmup and
  measured result is checked on all four GPUs against the NCCL reference.
  NaN/Inf and mismatches fail the process; tolerance is `1e-5 + 1e-5*abs(ref)`.
- The slow physical device is swapped to logical rank 3. The subcommunicator
  uses exactly `devs[0:3]`, excluding it. This is a single-process device mapping;
  it does not implement the team's future multi-process remap.
- Before generated or handwritten code, in-place `ncclReduceScatter` writes
  rank r's sum to `buffers[r] + r*chunk`. Non-straggler streams finish before
  switching from the subcommunicator to the full communicator. This work can
  overlap the slow GPU's delay. See [NVIDIA's in-place convention](https://github.com/NVIDIA/nccl/blob/master/docs/userguide/source/api/colls.rst).
- A clock64 spin kernel delays logical rank 3 by the same nominal duration for
  all algorithms. Cycles use the device's reported clock rate; actual delay may
  vary with GPU clocks. The upstream clock-based method is retained with a
  wider counter. No nominal delay is subtracted from measured runtime.
- Host steady-clock timing starts before delay injection and ends after all
  streams complete. It includes launch overhead, delay, ReduceScatter and the
  full collective. Buffer setup and host correctness copies are excluded.
  Algorithm order rotates across iterations to reduce ordering bias.
- The handwritten baseline calls the upstream helper verbatim and therefore
  includes its original internal synchronization and event behavior. It is a
  baseline, not a claim that the handwritten and generated launch paths are
  identical. Its event setup is amortized by warmup.

## Retained evidence and plots

`results/gpu-<UTC timestamp>/` contains raw case logs, commit, dirty-worktree
status, GPU/driver details and runtime versions. The raw output contains one
`RESULT` row per validated timing sample, plus reference and completion markers.
The entire `results/` directory is intended to be retained with the submission;
`logs/` is ignored by the original repo.

```bash
python3 harness/report.py results/gpu-*/b*.log --out results/summary --plots
```

The summary needs only Python's standard library. `--plots` additionally needs
matplotlib and produces PNG/PDF latency and speedup plots. Speedup is the median
of iteration-paired `nccl_ms / candidate_ms` within the **same case log**; timings
from different jobs are never paired. Raw timings, min/median/max and maximum
absolute error are retained in CSVs. `poster_results.txt` contains the measured
results panel. A missing timed NCCL baseline produces no speedup claim.

## Local checks (no GPU required)

```bash
python3 -m unittest discover -s harness -p 'test_*.py' -v
c++ -std=c++17 harness/test_validation.cpp -o /tmp/test_validation
/tmp/test_validation
python3 compiler/simulate.py ir/n4.json ir/n8.json ir/n16.json
bash -n scripts/sweep_harness.sh scripts/run_harness.sbatch
```

These checks do not establish CUDA compilation, NCCL execution, or GPU speedup.
The four-GPU harness does not complete the final two-node / eight-GPU milestone.

AI assistance: OpenAI Codex assisted with harness implementation, result
processing, documentation, and local validation. Retain actual GPU logs before
claiming measured correctness or performance.

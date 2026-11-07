# Test harness

The midpoint harness compares NCCL AllReduce, the handwritten n=4 StragglAR
implementation, and the generated n=4 code. It uses one process and four GPUs.
The generator and handwritten source are unchanged.

## Run

On Perlmutter, from the repo root:

```bash
source scripts/env.sh
make harness
sbatch --export=ALL,BIN=build/harness4 scripts/run_1node.sbatch 12582912 all 10 1
sbatch scripts/run_harness.sbatch  # full sweep
```

Inside an allocation:

```bash
build/harness4 bytes algorithm iterations delay_ms [warmups=5] [slow_gpu=3] [seed=5470]
```

Algorithms: `nccl`, `handwritten`, `generated`, or `all`. `stragglar` aliases
`handwritten`. Bytes must be positive and divisible by 12; the team's multiple
of 48 works. Delay `-1` means zero delay, but still runs ReduceScatter.

The sweep uses 12 KiB, 12 MiB and 48 MiB per GPU; delays 0/1/5 ms; slow GPUs 3/0;
five warmups and ten samples per algorithm. Override `SIZES`, `DELAYS`,
`STRAGGLERS`, `ITERS`, `WARMUPS`, `SEED`, or `OUT` in the job environment.
Raw logs and machine details are saved under `results/gpu-<timestamp>/`.

## Method

- Inputs vary by GPU, element and seed. Their binary fractions sum exactly in
  float32. Buffers reset before every run.
- NCCL is checked against a CPU sum. Every warmup and timed output is checked
  on all four GPUs against NCCL. NaN/Inf or an error above
  `1e-5 + 1e-5*abs(reference)` fails the run.
- The chosen slow GPU becomes logical rank 3. ReduceScatter uses only ranks
  0..2 and writes chunk r at `buffers[r] + r*chunkSize`. Those streams finish
  before switching communicators. This work overlaps the slow GPU's delay.
- The same clock64 delay runs on logical rank 3 for each algorithm. Delay is
  nominal because GPU clocks can vary. No delay is subtracted from timing.
- Host timing includes delay, ReduceScatter, communication and completion of
  every stream. Input setup and validation are excluded. Algorithm order rotates
  each iteration. The handwritten baseline retains its original internal syncs.

This is a single-node device remap, not the final multi-process implementation.
Small timing differences need repeated trials. These inputs do not establish
accuracy for arbitrary real-valued reductions.

## Results and checks

```bash
python3 harness/report.py results/gpu-*/b*.log --out results/summary --plots
python3 harness/poster.py results/summary/summary.csv results/summary/poster_results.pdf
python3 -m unittest discover -s harness -p 'test_*.py' -v
c++ -std=c++17 harness/test_validation.cpp -o /tmp/test_validation
/tmp/test_validation
python3 compiler/simulate.py ir/n4.json ir/n8.json ir/n16.json
```

Reports include raw samples, min/median/max latency, paired speedups and maximum
error. Speedup is the median of `NCCL_ms / candidate_ms` for matching iterations
in the same case. Incomplete logs are rejected. Select one sweep for plots and
the 14 x 10 inch panel. Plots need matplotlib; the panel needs reportlab.

Measured results and job evidence: [results/README.md](../results/README.md).
The host tests alone do not establish GPU correctness or performance.

AI assistance: Codex helped implement, test and document this work.

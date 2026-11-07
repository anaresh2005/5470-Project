# Midpoint plan

The midpoint poster **must be printed by Sun Nov 8 at the latest**.
We are behind the proposal timeline: the parser and IR (due Oct 31) and the launch script (due Nov 1) are not started.

Midpoint poster content (from the proposal): schedules for several n, the schedule-format and IR spec,
and generated code for n=4 running on one node and passing the harness.

## Deadlines

| Deadline | What must be done |
|---|---|
| **ASAP** | Anything marked *(blocking)* below — other people are waiting on it. |
| **Thu Nov 5** | Each person's part works on its own. |
| **Fri Nov 6** | n=4 runs end to end on one node and matches `ncclAllReduce`. |
| **Sat Nov 7** | Every poster panel done; final poster PDF exported. |
| **Sun Nov 8** | Poster printed (check the print shop's Sunday hours; print Saturday if possible). |

## Shared interfaces *(blocking, agree first)*

- [X] **IR format** (Wyatt proposes, everyone signs off). JSON with `n`, `straggler`, and a list of rounds,
      each a list of ops `{"src": int, "dst": int, "chunk": int, "op": "add" | "write"}`.
- [X] **Generated-code entry point** (David proposes, Aneesh signs off). For the midpoint, use the same
      single-process style as `stragglar/stragglar/allreduce_4GPU.cu` (one process, all GPUs, `ncclCommInitAll`), e.g.
      `void gen_stragglar_allreduce(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, int numRanks, size_t chunkSize)`.
      This keeps multi-process NCCL off the midpoint critical path; it only matters for 2 nodes.
      Final signature and calling rules: `gen/gen_stragglar.h`.

## Wyatt: schedules, parser, IR, simulator

By Thu Nov 5:
- [X] *(blocking)* Hand-write `ir/n4.json` from the n=4 schedule and push it, so David can start before the parser exists.
- [X] Run `synthesizer_pow2.py` for n = 4, 8, 16 and commit the outputs to `schedules/` (needs only `numpy`).
- [X] Spec of the schedule format (StragglerMatching / OneWayMatching / TwoWayMatching) and of the IR (`ir/README.md`).
- [X] Parser from schedule output to IR JSON; check it reproduces `ir/n4.json`.
- [X] CPU simulator (`compiler/simulate.py`) that checks the IR: no rank sends or receives twice in a round,
      nothing is summed twice, and every GPU ends with the full sum. Run it on n = 4, 8, 16.

By Sat Nov 7:
- [ ] Poster panel: problem, StragglAR background, correctness checking (simulator results).

After midpoint:
- [ ] Straggler remap (whichever GPU is slow becomes rank n−1).

## David: code generator

By Thu Nov 5:
- [ ] Read `stragglar_allreduce_helper` in `stragglar/stragglar/allreduce_4GPU.cu` (the hand-written n=4 version);
      work out how chunks map to buffer offsets and when scratch buffers are needed.
- [ ] Generator from IR JSON to C++/CUDA, one `ncclGroupStart`/`ncclGroupEnd` block per round,
      using the repo's `reduce_add` kernel for `add` ops. Start from Wyatt's hand-written `ir/n4.json`.
- [ ] Generated n=4 code compiles; diff it against the hand-written version and explain any differences.

By Sat Nov 7:
- [ ] Poster panel: compiler design and an example of generated code.

## Nathnael: build, launch, cluster

By Thu Nov 5:
- [ ] *(blocking)* Confirm all four of us can log in to Perlmutter and that the allocation has GPU hours.
- [X] *(blocking)* NCCL (module or build) plus a `Makefile`; compile and run the repo's `allreduce_4GPU.cu` on one node.
- [X] *(blocking)* Slurm script for 1 node / 4 GPUs that Aneesh and David can reuse.
- [X] Multi-process NCCL setup (one process per GPU, `ncclUniqueId` passed via MPI or a file) with a
      hand-written send/recv test on 4 GPUs. Not needed for the midpoint run, but needed for 2 nodes.

By Sat Nov 7:
- [ ] Poster panel: experimental setup and system details.
- [X] Stretch: the send/recv test on 8 GPUs across 2 nodes.

## Aneesh: test harness

By Thu Nov 5:
- [X] Harness adapted from the repo's timing code: buffer init (`fill_pattern`), `ncclAllReduce` as the reference,
      correctness check, timing loop.
- [X] ReduceScatter among ranks 0..n−2 before calling `gen_stragglar_allreduce` (see `gen/gen_stragglar.h`),
      as `stragglar_allreduce_delay` in `stragglar/stragglar/allreduce_4GPU.cu` does: a sub-communicator
      without the straggler, then `ncclReduceScatter`. Without it the result won't match `ncclAllReduce`.
- [X] Straggler delay injection on rank n−1 (the repo's `gpu_sleep_kernel`), with the same delay applied to
      `ncclAllReduce` and to our program.
- [X] Harness runs on Perlmutter with `ncclAllReduce` alone, using Nathnael's Slurm script; also run the repo's
      hand-written StragglAR through it as a baseline.

By Sat Nov 7:
- [X] Poster panel: results (correctness, and any n=4 timings vs. `ncclAllReduce`).

## Integration (everyone, by Fri Nov 6)

- [X] Generated n=4 program plus the ReduceScatter phase run in Aneesh's harness on 1 node and match `ncclAllReduce`.
- [ ] Fallback if the GPU run fails: the poster shows the simulator checking the IR for n = 4, 8, 16, plus the GPU run status.

## Aneesh results (Nov 7)

Perlmutter jobs 60968895 and 60968954 passed. All 540 timed runs across 18 cases
matched NCCL on every GPU, with zero maximum error. Generated code was slower
than NCCL in this sweep (0.315x–0.994x). Performance tuning and the final
8-GPU/two-node test remain.

Code and commands: `harness/README.md`. Logs, plots and results panel:
`results/gpu-20261107T173045Z/`.

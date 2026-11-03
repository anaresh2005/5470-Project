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

- [X] **IR format** (Wyatt proposes, everyone signs off). JSON with `n`, `num_chunks`, and a list of rounds,
      each a list of ops `{"src": int, "dst": int, "chunk": int, "op": "add" | "write"}`.
- [ ] **Generated-code entry point** (David proposes, Aneesh signs off). For the midpoint, use the same
      single-process style as `stragglar/stragglar/allreduce_4GPU.cu` (one process, all GPUs, `ncclCommInitAll`), e.g.
      `void gen_stragglar_allreduce(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, int numRanks, size_t chunkSize)`.
      This keeps multi-process NCCL off the midpoint critical path; it only matters for 2 nodes.

## Wyatt: schedules, parser, IR, simulator, ReduceScatter phase

By Thu Nov 5:
- [X] *(blocking)* Hand-write `ir/n4.json` from the n=4 schedule and push it, so David can start before the parser exists.
- [ ] Run `synthesizer_pow2.py` for n = 4, 8, 16 and commit the outputs to `schedules/` (needs only `numpy`).
- [ ] Spec of the schedule format (StragglerMatching / OneWayMatching / TwoWayMatching) and of the IR.
- [ ] Parser from schedule output to IR JSON; check it reproduces `ir/n4.json`.
- [ ] CPU simulator: every send has a matching receive in its round, no rank is used twice in a round,
      and the final buffers equal the true sum. Run it on n = 4, 8, 16.
- [ ] ReduceScatter phase before the schedule (sub-communicator without the straggler), reusing
      `stragglar/stragglar/reducescatter.cu`. Hand it to David to prepend to the generated code.

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
- [ ] *(blocking)* NCCL (module or build) plus a `Makefile`; compile and run the repo's `allreduce_4GPU.cu` on one node.
- [ ] *(blocking)* Slurm script for 1 node / 4 GPUs that Aneesh and David can reuse.
- [ ] Multi-process NCCL setup (one process per GPU, `ncclUniqueId` passed via MPI or a file) with a
      hand-written send/recv test on 4 GPUs. Not needed for the midpoint run, but needed for 2 nodes.

By Sat Nov 7:
- [ ] Poster panel: experimental setup and system details.
- [ ] Stretch: the send/recv test on 8 GPUs across 2 nodes.

## Aneesh: test harness

By Thu Nov 5:
- [ ] Harness adapted from the repo's timing code: buffer init (`fill_pattern`), `ncclAllReduce` as the reference,
      correctness check, timing loop.
- [ ] Straggler delay injection on rank n−1 (the repo's `gpu_sleep_kernel`), with the same delay applied to
      `ncclAllReduce` and to our program.
- [ ] Harness runs on Perlmutter with `ncclAllReduce` alone, using Nathnael's Slurm script; also run the repo's
      hand-written StragglAR through it as a baseline.

By Sat Nov 7:
- [ ] Poster panel: results (correctness, and any n=4 timings vs. `ncclAllReduce`).

## Integration (everyone, by Fri Nov 6)

- [ ] Wyatt's simulator passes the generated n=4 program before the GPU run.
- [ ] Generated n=4 program plus the ReduceScatter phase run in Aneesh's harness on 1 node and match `ncclAllReduce`.
- [ ] Fallback if the GPU run fails: the poster shows the simulator checking the generated program, plus the GPU run status.

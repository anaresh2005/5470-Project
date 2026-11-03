# Midpoint plan (Nov 2 – Nov 8)

The midpoint poster **must be printed by Sun Nov 8 at the latest**. That means the final poster PDF
is done **Sat Nov 7**, and the n=4 pipeline has to work end to end by **Fri Nov 6**.
We are behind the proposal timeline: the parser and IR (due Oct 31) and the launch script (due Nov 1) are not started.

Midpoint poster content (from the proposal): schedules for several n, the schedule-format and IR spec,
and generated code for n=4 running on one node and passing the harness.

## Mon Nov 2: agree on the interfaces first

Everyone works against these from day one, so nobody waits on anyone else.

- [ ] **IR format** (Wyatt proposes, everyone signs off). JSON with a list of rounds, each a list of
      ops `{"src": int, "dst": int, "chunk": int, "op": "add" | "copy"}`, plus `n` and `num_chunks`.
- [ ] **Generated-code entry point** (David proposes, Aneesh signs off). For the midpoint, use the same
      single-process style as `stragglar/stragglar/allreduce_4GPU.cu` (one process, all GPUs,
      `ncclCommInitAll`), e.g.
      `void gen_stragglar_allreduce(float** d_buffers, float** d_tempbufs, int* devs, cudaStream_t* streams, ncclComm_t* comms, int numRanks, size_t chunkSize)`.
      This keeps multi-process NCCL off the midpoint critical path; it only matters for 2 nodes.
- [ ] Everyone: get the StragglAR code. It needs only `numpy` to run the synthesizer:
      `pip install numpy && python stragglar/stragglar/synthesizer_pow2.py 4`.

## Wyatt: schedules, parser, IR, simulator, ReduceScatter phase

- [ ] **Mon Nov 2:** run `synthesizer_pow2.py` for n = 4, 8, 16 and commit the outputs to `schedules/`.
- [ ] **Mon Nov 2:** hand-write `ir/n4.json` from the n=4 schedule and push it, so David can start before the parser exists.
- [ ] **Tue Nov 3:** write the spec of the schedule format (StragglerMatching / OneWayMatching / TwoWayMatching) and of the IR.
- [ ] **Tue Nov 3:** parser from schedule output to IR JSON. Check that it reproduces `ir/n4.json` exactly.
- [ ] **Wed Nov 4:** CPU simulator: every send has a matching receive in its round, no rank is used twice in a round,
      and the final buffers equal the true sum. Run it on n = 4, 8, 16.
- [ ] **Thu Nov 5:** ReduceScatter phase before the schedule (sub-communicator without the straggler),
      reusing `stragglar/stragglar/reducescatter.cu`. Hand it to David to prepend to the generated code.
- [ ] **Fri Nov 6:** integration (see below).
- [ ] **Sat Nov 7:** poster panel: problem, StragglAR background, correctness checking (simulator results).
- [ ] *After midpoint:* straggler remap (whichever GPU is slow becomes rank n−1).

## David: code generator

- [ ] **Mon Nov 2 – Tue Nov 3:** read `stragglar_allreduce_helper` in `stragglar/stragglar/allreduce_4GPU.cu`
      (the hand-written n=4 version). Write down how chunks map to buffer offsets and when scratch buffers are needed.
- [ ] **Tue Nov 3 – Wed Nov 4:** generator from IR JSON to C++/CUDA, one `ncclGroupStart`/`ncclGroupEnd` block per round,
      using the repo's `reduce_add` kernel for `add` ops. Start from Wyatt's hand-written `ir/n4.json`.
- [ ] **Thu Nov 5:** generated n=4 code compiles; diff it against the hand-written version and explain any differences.
- [ ] **Fri Nov 6:** integration (see below).
- [ ] **Sat Nov 7:** poster panel: compiler design and an example of generated code.

## Nathnael: build, launch, cluster

- [ ] **Mon Nov 2 (urgent):** confirm all four of us can log in to Perlmutter and that the allocation has GPU hours.
- [ ] **Tue Nov 3:** NCCL (module or build) plus a `Makefile`; compile and run the repo's `allreduce_4GPU.cu` on one node.
- [ ] **Wed Nov 4:** Slurm script for 1 node / 4 GPUs that Aneesh and David can reuse.
- [ ] **Thu Nov 5:** multi-process NCCL setup (one process per GPU, `ncclUniqueId` passed via MPI or a file)
      with a hand-written send/recv test on 4 GPUs. Not needed for the midpoint run, but needed for 2 nodes.
- [ ] **Fri Nov 6:** integration (see below). Stretch: the send/recv test on 8 GPUs across 2 nodes.
- [ ] **Sat Nov 7:** poster panel: experimental setup and system details.

## Aneesh: test harness

- [ ] **Mon Nov 2 – Tue Nov 3:** harness adapted from the repo's timing code: buffer init (`fill_pattern`),
      `ncclAllReduce` as the reference, correctness check, timing loop.
- [ ] **Wed Nov 4:** straggler delay injection on rank n−1 (the repo's `gpu_sleep_kernel`), with the same delay
      applied to `ncclAllReduce` and to our program.
- [ ] **Thu Nov 5:** harness runs on Perlmutter with `ncclAllReduce` alone, using Nathnael's Slurm script.
      Also run the repo's hand-written StragglAR through it as a baseline.
- [ ] **Fri Nov 6:** integration (see below).
- [ ] **Sat Nov 7:** poster panel: results (correctness, and any n=4 timings we have vs. `ncclAllReduce`).

## Fri Nov 6: integration day (everyone)

- [ ] Wyatt's simulator passes the generated n=4 program before the GPU run.
- [ ] Generated n=4 program and ReduceScatter phase run in Aneesh's harness on 1 node and match `ncclAllReduce`.
- [ ] Fallback if the GPU run fails: the poster shows the simulator checking the generated program,
      plus the GPU run status.

## Sat Nov 7 – Sun Nov 8: poster

- [ ] **Sat Nov 7, noon:** each panel's content in the shared poster file.
- [ ] **Sat Nov 7, evening:** full team review; export the final PDF.
- [ ] **Sun Nov 8 at the latest:** print (check the print shop's Sunday hours in advance; print Sat if possible).
- [ ] **Mon Nov 9:** midpoint session.

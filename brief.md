# **A compiler from StragglAR schedules to NCCL send/receive programs**

* **Team:** Team 31 (Chesterjesters) — Wyatt Sell, David Valarezo, Nathnael Tesfaw, Aneesh Naresh

* **Scope:** 8 GPUs across two nodes (NCCL over Slingshot)

* **Advisor:** Arjun Devraj — adevraj@cs.cornell.edu — GitHub @arjundevraj

* **Compute:** two Perlmutter nodes (8×A100)

## **Background**

AllReduce is the collective that synchronizes gradients in data-parallel training: every GPU contributes a buffer and every GPU ends with the element-wise sum. The standard algorithms (ring, tree) assume all GPUs are ready at the same time. In real clusters one GPU is often late — a straggler — because of thermal throttling, a slow host, contention on a link, or a preempted kernel. A ring AllReduce cannot finish until the straggler joins, so every other GPU idles for the whole delay.

StragglAR is an AllReduce algorithm designed for exactly this case. The GPUs that are ready do most of the reduction among themselves while the straggler is still late, and the straggler is folded in with a short final phase once it arrives, so the collective finishes sooner than ring does after the straggler shows up. StragglAR’s schedule synthesizer takes the number of GPUs, n, assumes the straggler is the last rank (n−1), and emits a schedule of chunk transfers — which rank sends which chunk of the buffer to which rank in which step — in a custom text format of its own.

What the schedule does not do is run. To execute it on a real cluster, every transfer has to be turned into point-to-point NCCL operations (ncclSend/ncclRecv, grouped per step so they do not deadlock), with the local reductions in between. Today that translation is done by hand. The question is whether a compiler can treat the synthesizer’s output as an intermediate representation and generate the NCCL program automatically, so that any schedule the synthesizer produces can be dropped into a collective communication library.

## **Objective**

* Run the existing StragglAR code to produce schedules for any power-of-two n.

* Build a compiler — either a new tool that consumes the synthesizer’s output, or an extension of the StragglAR codebase — that translates a schedule into a program of ncclSend/ncclRecv calls (plus the local reduction steps) that executes the algorithm on a multi-GPU cluster.

* Show that the generated code compiles and runs to completion without deadlock on 8 GPUs across two Perlmutter nodes (NCCL over Slingshot), and that its result matches a reference AllReduce.

## **Example deliverables**

* Schedules generated for several power-of-two values of n.

* The compiler, with its intermediate representation documented.

* A test showing the generated program compiles and runs without deadlock on the target GPU count, with a correctness check against NCCL’s own AllReduce and timing under an injected straggler delay.

## **Code and tools**

* StragglAR: https\://github.com/arjundevraj/stragglar

* NCCL point-to-point primitives (ncclSend, ncclRecv, ncclGroupStart/ncclGroupEnd)


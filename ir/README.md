# Schedule format and IR

## Schedule (input)

`stragglar/stragglar/synthesizer_pow2.py <n>` prints one block per round. The line after `Round k` lists that round's matchings:

| Matching | Text | Meaning |
|---|---|---|
| StragglerMatching | `a <-> b, chunk_id: c` | `a` and the straggler `b` exchange chunk `c` and both add it |
| OneWayMatching | `a -> b, chunk_id: c` | `a` sends its finished chunk `c` to `b` |
| TwoWayMatching | `a -> b, chunk_id: x; b -> a, chunk_id: y` | two OneWayMatchings in opposite directions |

The `GPU k chunks [...]` lines show which finished chunks each GPU holds after the round.

## IR (output)

Generate with `python3 compiler/parser.py schedules/n<n>.txt ir/n<n>.json`; check with `python3 compiler/simulate.py ir/n<n>.json`.

```json
{
  "version": 1,
  "n": 4,
  "straggler": 3,
  "source": "schedules/n4.txt",
  "rounds": [
    [
      {"src": 0, "dst": 3, "chunk": 0, "op": "add"},
      {"src": 3, "dst": 0, "chunk": 0, "op": "add"}
    ]
  ]
}
```

- **Chunks.** Each buffer is split into `n - 1` chunks; chunk `c` starts at `c * chunkSize`.
- **Starting state.** Before round 0, ranks `0..n-2` run a ReduceScatter among themselves, so rank `r` holds chunk `r` summed over every GPU except the straggler.
- **Ops.** Each op is one transfer: an `ncclSend` on `src` and an `ncclRecv` on `dst`.
  - `add`: `dst[chunk] += src[chunk]`. Receive into a scratch buffer, then `reduce_add`. Only used for exchanges with the straggler.
  - `write`: `dst[chunk] = src[chunk]`. Receive straight into the buffer. Only used for chunks that are already fully summed.
- **Rounds.** Each round is one `ncclGroupStart`/`ncclGroupEnd` block. All ops read values from the start of the round, and each rank sends at most once and receives at most once per round, so one scratch buffer per GPU is enough.
- **Straggler.** Always `n - 1` for now; the straggler remap will change it.

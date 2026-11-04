import sys
from pathlib import Path
import json

# Runs an IR file on pretend GPUs. Each chunk is tracked as the set of GPUs
# whose data has been summed into it, so a correct AllReduce ends with every
# chunk on every GPU equal to {0, ..., n-1}.

args = sys.argv[1:]

if len(args) == 0:
    print("Usage: python simulate.py <ir_file> [<ir_file> ...]")
    sys.exit(1)


def simulate(ir):
    n = ir["n"]
    straggler = ir["straggler"]
    everyone = set(range(n))

    # After the ReduceScatter among the non-stragglers, rank r holds chunk r
    # summed over every GPU except the straggler. Every other chunk holds
    # only the GPU's own data.
    bufs = [[{gpu} for _ in range(n - 1)] for gpu in range(n)]
    for r in range(n - 1):
        bufs[r][r] = everyone - {straggler}

    for i, ops in enumerate(ir["rounds"]):
        srcs = [op["src"] for op in ops]
        dsts = [op["dst"] for op in ops]
        for rank in set(srcs):
            if srcs.count(rank) > 1:
                return f"round {i}: GPU {rank} sends more than once"
        for rank in set(dsts):
            if dsts.count(rank) > 1:
                return f"round {i}: GPU {rank} receives more than once"

        # every op in a round reads the values from the start of the round
        before = [[set(chunk) for chunk in buf] for buf in bufs]
        for op in ops:
            src, dst, chunk = op["src"], op["dst"], op["chunk"]
            if not (0 <= src < n and 0 <= dst < n and 0 <= chunk < n - 1) or src == dst:
                return f"round {i}: invalid op {op}"
            sent = before[src][chunk]
            if op["op"] == "add":
                overlap = sent & before[dst][chunk]
                if overlap:
                    return f"round {i}: GPU {src} -> GPU {dst} chunk {chunk} adds GPUs {sorted(overlap)} twice"
                bufs[dst][chunk] = before[dst][chunk] | sent
            elif op["op"] == "write":
                if sent != everyone:
                    return f"round {i}: GPU {src} -> GPU {dst} writes chunk {chunk} before it is fully summed"
                bufs[dst][chunk] = set(sent)
            else:
                return f"round {i}: unknown op type {op['op']!r}"

    for rank, buf in enumerate(bufs):
        for chunk, summed in enumerate(buf):
            if summed != everyone:
                return f"end: GPU {rank} chunk {chunk} is missing GPUs {sorted(everyone - summed)}"
    return None


failed = False
for source_file in args:
    file_path = Path.cwd() / source_file
    if not file_path.exists():
        print(f"Error: File '{source_file}' does not exist.")
        sys.exit(1)

    ir = json.loads(file_path.read_text())
    error = simulate(ir)
    if error:
        failed = True
        print(f"FAIL {source_file}: {error}")
    else:
        print(f"PASS {source_file}: n={ir['n']}, {len(ir['rounds'])} rounds, all GPUs hold the full sum")

sys.exit(1 if failed else 0)

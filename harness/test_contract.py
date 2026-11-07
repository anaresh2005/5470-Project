"""Numeric host checks of the harness's ReduceScatter + n=4 schedule contract.

These do NOT execute CUDA or NCCL. They cover layout, physical-device remapping,
and omission of the prerequisite phase, using nonuniform signed inputs.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pattern(gpu, index, seed):
    x = (index ^ ((gpu + 1) * 0x9e3779b9) ^ seed) & 0xffffffff
    x ^= x >> 16
    x = (x * 0x7feb352d) & 0xffffffff
    x ^= x >> 15
    return (x % 2049 - 1024) / 256


def execute(chunk, slow_gpu, seed, prephase=True):
    devs = list(range(4))
    devs[slow_gpu], devs[3] = devs[3], devs[slow_gpu]
    initial = [[pattern(devs[r], i, seed) for i in range(3 * chunk)] for r in range(4)]
    expected = [sum(initial[r][i] for r in range(4)) for i in range(3 * chunk)]
    bufs = [b[:] for b in initial]
    if prephase:
        for r in range(3):
            for i in range(r * chunk, (r + 1) * chunk):
                bufs[r][i] = sum(initial[j][i] for j in range(3))
    ir = json.loads((ROOT / 'ir/n4.json').read_text())
    for ops in ir['rounds']:
        before = [b[:] for b in bufs]
        for op in ops:
            for i in range(op['chunk'] * chunk, (op['chunk'] + 1) * chunk):
                src = before[op['src']][i]
                bufs[op['dst']][i] = src + before[op['dst']][i] if op['op'] == 'add' else src
    return bufs, expected


class ContractTests(unittest.TestCase):
    def test_all_physical_stragglers_nonuniform_chunks_and_seeds(self):
        for chunk in (1, 7, 257):
            for slow in range(4):
                for seed in (0, 5470, 0xffffffff):
                    with self.subTest(chunk=chunk, slow=slow, seed=seed):
                        buffers, expected = execute(chunk, slow, seed)
                        self.assertTrue(all(b == expected for b in buffers))

    def test_missing_reducescatter_is_detected(self):
        buffers, expected = execute(7, 3, 5470, prephase=False)
        self.assertTrue(any(b != expected for b in buffers))

if __name__ == '__main__':
    unittest.main()

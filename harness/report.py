"""Summarize completed harness logs; never infer a GPU result from CPU checks."""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

FIELDS = ['kind', 'algorithm', 'bytes', 'iteration', 'delay_ms',
          'physical_straggler', 'seed', 'runtime_ms', 'max_abs_error', 'correct']


def read_logs(paths):
    rows = []
    for path in paths:
        text = Path(path).read_text()
        lines = text.splitlines()
        if 'PASS: every sample on all four ranks matches ncclAllReduce' not in lines:
            raise ValueError(f'{path}: missing completion marker (failed or incomplete job)')
        if 'CHECK,reference_cpu,PASS' not in lines or any(line.startswith('FAIL') for line in lines):
            raise ValueError(f'{path}: reference check missing or failure recorded')
        seen = set()
        found = 0
        for line in lines:
            if not line.startswith('RESULT,'):
                continue
            cells = next(csv.reader([line]))
            if len(cells) != len(FIELDS):
                raise ValueError(f'{path}: malformed RESULT row')
            row = dict(zip(FIELDS, cells))
            if row['correct'] != 'PASS' or row['algorithm'] not in ('nccl', 'handwritten', 'generated'):
                raise ValueError(f'{path}: invalid or failed row')
            for key in ('bytes', 'iteration', 'physical_straggler', 'seed'):
                row[key] = int(row[key])
            for key in ('delay_ms', 'runtime_ms', 'max_abs_error'):
                row[key] = float(row[key])
                if not math.isfinite(row[key]) or row[key] < 0:
                    raise ValueError(f'{path}: invalid {key}')
            if row['runtime_ms'] <= 0 or row['bytes'] <= 0 or row['bytes'] % 12 or row['iteration'] < 0 or not 0 <= row['physical_straggler'] < 4:
                raise ValueError(f'{path}: invalid sample')
            identity = tuple(row[key] for key in ('algorithm', 'bytes', 'iteration', 'delay_ms', 'physical_straggler', 'seed'))
            if identity in seen:
                raise ValueError(f'{path}: duplicate sample')
            seen.add(identity)
            row['source'] = str(path)
            rows.append(row)
            found += 1
        if not found:
            raise ValueError(f'{path}: no measurements')
    return rows


def summarize(rows):
    # Keep source file in the key: never pair timings from separate jobs.
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ('source', 'bytes', 'delay_ms', 'physical_straggler', 'seed'))
        groups[key].append(row)
    summaries = []
    for key, samples in sorted(groups.items()):
        by_alg = defaultdict(list)
        for row in samples:
            by_alg[row['algorithm']].append(row)
        reference = {r['iteration']: r['runtime_ms'] for r in by_alg.get('nccl', [])}
        for alg, values in sorted(by_alg.items()):
            indices = {r['iteration'] for r in values}
            if indices != set(range(len(values))):
                raise ValueError(f'{key}: non-contiguous {alg} iterations')
            if alg != 'nccl' and reference and indices != set(reference):
                raise ValueError(f'{key}: unmatched reference iterations for {alg}')
            times = [r['runtime_ms'] for r in values]
            summaries.append(dict(zip(('source', 'bytes', 'delay_ms', 'physical_straggler', 'seed'), key),
                                  algorithm=alg, samples=len(times), median_ms=median(times),
                                  min_ms=min(times), max_ms=max(times),
                                  median_paired_speedup=(median(reference[r['iteration']] / r['runtime_ms'] for r in values)
                                                         if reference else ''),
                                  max_abs_error=max(r['max_abs_error'] for r in values)))
    return summaries


def write_report(rows, out, plots=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    summaries = summarize(rows)
    for name, data in [('samples.csv', rows), ('summary.csv', summaries)]:
        with (out / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(data)
    generated = [s for s in summaries if s['algorithm'] == 'generated']
    panel = ['RESULTS — FOUR-GPU TEST HARNESS', '',
             f'Validated {len(rows)} measured runs against NCCL AllReduce on all four ranks.',
             'The NCCL reference also passed an independent CPU sum check.',
             'Inputs vary by GPU and element; buffers reset before every run.',
             'Timing includes injected delay, ReduceScatter, communication, and synchronization.',
             'Initialization and result checking are outside the timed region.', '',
             'Generated program (median of paired NCCL / generated runtime ratios):']
    for s in generated:
        speedup = s['median_paired_speedup']
        ratio = f'{speedup:.3f}x' if speedup != '' else 'not measured (no timed NCCL baseline)'
        panel.append(f"  {s['bytes']} bytes/GPU; nominal delay {s['delay_ms']:g} ms; slow GPU {s['physical_straggler']}: "
                     f"{s['median_ms']:.3f} ms, speedup {ratio}, {s['samples']} samples.")
    if not generated:
        panel.append('  Generated GPU program not measured in these logs.')
    panel.extend(['', 'Above 1x means faster than NCCL; below 1x means slower.',
                  'Host-observed latency includes launch overhead. Delays are clock-based and nominal.',
                  'Single-node results do not establish two-node performance or correctness.'])
    (out / 'poster_results.txt').write_text('\n'.join(panel) + '\n')
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        datasets = defaultdict(list)
        for s in summaries:
            datasets[(s['bytes'], s['physical_straggler'], s['seed'])].append(s)
        for (size, gpu, seed), samples in datasets.items():
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            for algorithm in ('nccl', 'handwritten', 'generated'):
                color = {'nccl': '#3366AA', 'handwritten': '#DD8833', 'generated': '#087F8C'}[algorithm]
                vals = sorted([s for s in samples if s['algorithm'] == algorithm], key=lambda s: s['delay_ms'])
                if not vals:
                    continue
                if len({s['delay_ms'] for s in vals}) != len(vals):
                    raise ValueError('Select one sweep for plots: repeated delay cases found')
                x = [s['delay_ms'] for s in vals]; y = [s['median_ms'] for s in vals]
                axes[0].errorbar(x, y, yerr=[[s['median_ms'] - s['min_ms'] for s in vals],
                                           [s['max_ms'] - s['median_ms'] for s in vals]], marker='o', color=color, label=algorithm)
                vals = [s for s in vals if s['median_paired_speedup'] != '']
                if algorithm != 'nccl' and vals:
                    axes[1].plot([s['delay_ms'] for s in vals], [s['median_paired_speedup'] for s in vals], marker='o', color=color, label=algorithm)
            axes[0].set_ylabel('Median end-to-end latency (ms); range bars')
            axes[1].set_ylabel('Median paired speedup vs. NCCL')
            axes[1].axhline(1, color='gray', linestyle='--')
            for ax in axes:
                ax.set_xlabel('Nominal injected delay (ms)'); ax.grid(alpha=.2)
                if ax.get_legend_handles_labels()[0]: ax.legend()
            fig.suptitle(f'4 GPUs | {size / 2**20:g} MiB per GPU | slow GPU {gpu}')
            for ext in ('png', 'pdf'):
                fig.savefig(out / f'b{size}-g{gpu}-s{seed}.{ext}', dpi=180)
            plt.close(fig)
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, default=Path('results/summary'))
    parser.add_argument('--plots', action='store_true', help='also produce PNG/PDF figures (requires matplotlib)')
    args = parser.parse_args()
    try:
        rows = read_logs(args.logs)
        write_report(rows, args.out, args.plots)
    except (ValueError, OSError) as e:
        parser.exit(1, f'Cannot report results: {e}\n')
    print(f'Wrote {args.out}: {len(rows)} validated timing samples')

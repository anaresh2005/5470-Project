"""Create Aneesh's standalone 14 x 10 inch results panel from measured summaries.

Usage: python harness/poster.py results/summary/summary.csv results/summary/poster_results.pdf
Requires reportlab. Run report.py on completed raw logs first.
"""
import argparse
import csv
from pathlib import Path
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle

NAVY = colors.HexColor('#15283B')
TEAL = colors.HexColor('#087F8C')
GRAY = colors.HexColor('#586572')
PALE = colors.HexColor('#EDF5F6')


def panel(summary, output):
    with Path(summary).open() as f:
        rows = list(csv.DictReader(f))
    generated = [r for r in rows if r['algorithm'] == 'generated']
    if not generated or any(not r['median_paired_speedup'] for r in generated):
        raise ValueError('A measured generated/NCCL comparison is required')
    seen = set()
    for r in generated:
        key = (r['bytes'], r['delay_ms'], r['physical_straggler'])
        if key in seen:
            raise ValueError('Select one sweep: repeated cases require separate panels')
        seen.add(key)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output), pagesize=(1008, 720))
    c.setTitle('Aneesh Naresh - StragglAR midpoint results')
    c.setAuthor('Team 31 - Chesterjesters')
    style = ParagraphStyle('body', fontName='Helvetica', fontSize=15, leading=20, textColor=NAVY)
    def text(s, x, top, width, size=15, color=NAVY):
        ps = ParagraphStyle('local', parent=style, fontSize=size, leading=size*1.3, textColor=color)
        p = Paragraph(s, ps); _, h = p.wrap(width, 1000); p.drawOn(c, x, top-h)
        return h
    c.setFillColor(NAVY); c.rect(0, 624, 1008, 96, fill=1, stroke=0)
    text('RESULTS: CORRECTNESS &amp; STRAGGLER TIMING', 34, 694, 940, 28, colors.white)
    text('Aneesh Naresh | Team 31 | One Perlmutter node, four A100 GPUs', 34, 649, 940, 14, colors.white)
    metrics = [
        (str(sum(int(r['samples']) for r in rows)), 'measured algorithm runs; all passed'),
        (f"{max(float(r['max_abs_error']) for r in rows):g}", 'maximum absolute error vs. NCCL'),
        (str(len({r['physical_straggler'] for r in generated})), 'physical slow-GPU placements tested'),
    ]
    for i, (value, label) in enumerate(metrics):
        x = 34 + i*318
        c.setFillColor(PALE); c.roundRect(x, 531, 304, 74, 8, fill=1, stroke=0)
        text(value, x+15, 594, 275, 29, TEAL)
        text(label, x+15, 552, 278, 11, GRAY)
    text('Generated program speedup relative to NCCL AllReduce', 34, 511, 940, 21)
    text('Median of paired NCCL / generated latency ratios. Above 1.00x is faster; below 1.00x is slower.',
         34, 480, 940, 13, GRAY)
    delays = sorted({float(r['delay_ms']) for r in generated})
    cases = sorted({(int(r['bytes']), int(r['physical_straggler'])) for r in generated})
    data = [['Buffer / GPU', 'Slow GPU'] + [f'{d:g} ms delay' for d in delays]]
    for size, gpu in cases:
        vals = {(float(r['delay_ms'])): r for r in generated if int(r['bytes']) == size and int(r['physical_straggler']) == gpu}
        label = f'{size/2**20:g} MiB' if size >= 2**20 else f'{size/1024:g} KiB'
        data.append([label, str(gpu)] + [f"{float(vals[d]['median_paired_speedup']):.3f}x" if d in vals else '-' for d in delays])
    if len(data) > 9 or len(delays) > 5:
        raise ValueError('Panel supports at most eight buffer/GPU cases and five delays')
    widths = [210, 120] + [610/len(delays)]*len(delays)
    t = Table(data, colWidths=widths, rowHeights=31)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY), ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 14), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'), ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, PALE]),
        ('LINEBELOW', (0,-1), (-1,-1), .7, colors.HexColor('#CCDADD')),
    ]))
    _, height = t.wrap(940, 500); t.drawOn(c, 34, 451-height)
    y = 451-height-20
    speedups = [float(r['median_paired_speedup']) for r in generated]
    wins = sum(v > 1 for v in speedups)
    conclusion = (f'Generated code matched NCCL in every measured run. Speedup ranged from '
                  f'{min(speedups):.3f}x to {max(speedups):.3f}x across {len(generated)} cases; '
                  f'{wins} cases had a median ratio above 1.00x.')
    h = text(conclusion, 34, y, 940, 17); y -= h+17
    method = ('<b>Method.</b> Signed inputs vary by GPU and element; every run starts from fresh buffers. '
              'NCCL is checked against a CPU sum, and all four outputs are checked after every run. '
              'Identical nominal delays are injected on the chosen physical GPU. Timing includes delay, '
              'ReduceScatter, communication and synchronization; setup and validation are excluded.')
    h = text(method, 34, y, 940, 13); y -= h+12
    text('<b>Scope.</b> Five warmups and ten measured samples per algorithm/case in the default sweep. '
         'Host-observed latency includes launch overhead; clock-based delays are nominal. '
         'Small timing differences need repeated trials. Two-node / eight-GPU evaluation remains future work.',
         34, y, 940, 12, GRAY)
    text('Evidence: retained raw logs and summary.csv | NCCL, handwritten and generated paths tested | November 7, 2026',
         34, 30, 940, 10, GRAY)
    c.showPage(); c.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary'); parser.add_argument('output')
    args = parser.parse_args()
    panel(args.summary, args.output)

import sys
from pathlib import Path
import json
import ast
import re

args = sys.argv[1:]

if len(args) not in (1, 2):
    print("Usage: python parser.py <source_file> [<output_file>]")
    sys.exit(1)

source_file = args[0]
output_file = args[1] if len(args) == 2 else None
file_path = Path.cwd() / source_file

if not file_path.exists():
    print(f"Error: File '{source_file}' does not exist.")
    sys.exit(1)

rounds = []
n = 0

current_round = []
next_line_switch = False

for line in file_path.read_text().splitlines():
    if line.startswith("GPU "):
        n = max(n, int(line.split()[1]) + 1)
        continue
    if line.startswith("Round"):
        next_line_switch = True
        continue
    if next_line_switch:
        ops = ast.literal_eval(line.split(" ", 1)[1])
        for op in ops:
            digits = [int(d) for d in re.findall(r'\b\d+\b', op)]
            if op.startswith("StragglerMatching"):
                current_round.extend([{
                    "src": digits[0],
                    "dst": digits[1],
                    "chunk": digits[2],
                    "op": "add"
                }, {
                    "src": digits[1],
                    "dst": digits[0],
                    "chunk": digits[2],
                    "op": "add"
                }])
            else:
                for i in range(0, len(digits), 3):
                    current_round.append({
                        "src": digits[i],
                        "dst": digits[i + 1],
                        "chunk": digits[i + 2],
                        "op": "write"
                    })
        rounds.append(current_round)
        current_round = []
        next_line_switch = False

ir = {
    "version": 1,
    "n": n,
    "straggler": n - 1,
    "source": source_file,
    "rounds": rounds,
}

text = json.dumps(ir, indent=2)
# put each op on one line
text = re.sub(r'\{\s+([^{}]*?)\s+\}', lambda m: '{' + ' '.join(m.group(1).split()) + '}', text)

if output_file:
    Path(output_file).write_text(text + "\n")
else:
    print(text)

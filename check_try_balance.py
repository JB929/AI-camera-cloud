import sys
from collections import deque

path = sys.argv[1] if len(sys.argv) > 1 else "src/core/multi_camera_detector.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

try_positions = []
for i, L in enumerate(lines, start=1):
    s = L.strip()
    if s.startswith("try:"):
        try_positions.append(("try", i))
    elif s.startswith("except") or s.startswith("except "):
        try_positions.append(("except", i))
    elif s.startswith("finally:"):
        try_positions.append(("finally", i))

# compute running balance
balance = 0
imbalances = []
for kind, ln in try_positions:
    if kind == "try":
        balance += 1
    else:
        balance -= 1
    imbalances.append((ln, kind, balance))

print("Last few try/except/finally markers (line, kind, running_balance):")
for x in imbalances[-40:]:
    print(x)

if balance != 0:
    print("\n*** Imbalance detected: final running_balance =", balance)
    print("Check the earliest 'try' without a matching except/finally above line where the parser failed.")
else:
    print("\ntry/except/finally counts appear balanced (balance==0).")

# also show 40 lines around the location where Python reported the syntax error (line ~276)
err_line = 276
start = max(0, err_line-20)
end = min(len(lines), err_line+20)
print(f"\n--- Source context around line {err_line} ---")
for i in range(start, end):
    print(f"{i+1:4}: {lines[i].rstrip()}")

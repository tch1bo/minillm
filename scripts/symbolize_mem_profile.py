import linecache
import pickle
import sys
from collections import defaultdict

if len(sys.argv) != 2:
    raise ValueError(
        f"expected one argument in argv (the path to the mem profile). Got: {sys.argv[1:]}"
    )

with open(sys.argv[1], "rb") as f:
    snap = pickle.load(f)

by_frame: dict[tuple, int] = defaultdict(int)
for seg in snap["segments"]:
    for block in seg["blocks"]:
        if block["state"] == "active_allocated":
            frames = block.get("frames", [])
            key = next(
                (
                    (fr["filename"], fr["line"])
                    for fr in frames
                    if "site-packages" not in fr["filename"]
                ),
                None,
            )
            if key is not None:
                by_frame[key] += block["size"]

for loc, size in sorted(by_frame.items(), key=lambda kv: -kv[1])[:15]:
    if loc is None:
        print(f"{size/2**20:8.1f} MB  torch-internal")
        continue
    filename, lineno = loc
    src = linecache.getline(filename, lineno).strip() or "<source unavailable>"
    print(f"{size/2**20:8.1f} MB  {filename}:{lineno}")
    print(f"{'':12s}{src}")

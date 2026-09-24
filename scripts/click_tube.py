#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
from reconstruction.image.clicking import click_polyline

p = argparse.ArgumentParser()
p.add_argument("image"); p.add_argument("-o", "--output")
a = p.parse_args()
points = click_polyline(a.image)
out = a.output or str(Path(a.image).with_name(Path(a.image).stem + "_tube.csv"))
Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["index", "u", "v"])
    for i, (u,v) in enumerate(points): w.writerow([i, u, v])
print(f"Wrote {out} ({len(points)} points)")

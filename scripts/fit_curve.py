#!/usr/bin/env python3
import argparse, csv
import numpy as np
from reconstruction.geometry.curves import fit_bspline

def read_points(path):
    with open(path, newline="", encoding="utf-8") as f:
        return np.asarray([[float(r["x"]),float(r["y"]),float(r["z"])]
                           for r in csv.DictReader(f)])

p = argparse.ArgumentParser()
p.add_argument("input"); p.add_argument("-o","--output",required=True)
p.add_argument("--smoothing",type=float,default=0.0)
p.add_argument("--samples",type=int,default=200)
a = p.parse_args()

X = read_points(a.input)
curve,_ = fit_bspline(X,a.smoothing,a.samples)
with open(a.output,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["index","x","y","z"])
    for i,pnt in enumerate(curve): w.writerow([i,*pnt])
print(f"Wrote {a.output} ({len(curve)} samples)")

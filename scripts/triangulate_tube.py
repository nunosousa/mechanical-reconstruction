#!/usr/bin/env python3
import argparse, csv
import numpy as np
from reconstruction.camera.pose import load_camera
from reconstruction.camera.projection import projection_matrix
from reconstruction.reconstruction.triangulation import triangulate

def read_tube(path):
    with open(path, newline="", encoding="utf-8") as f:
        return np.asarray([[float(r["u"]), float(r["v"])] for r in csv.DictReader(f)])

p = argparse.ArgumentParser()
p.add_argument("camera1"); p.add_argument("tube1")
p.add_argument("camera2"); p.add_argument("tube2")
p.add_argument("-o", "--output", required=True)
a = p.parse_args()

c1, c2 = load_camera(a.camera1), load_camera(a.camera2)
x1, x2 = read_tube(a.tube1), read_tube(a.tube2)
if len(x1) != len(x2):
    raise ValueError("Tube files must contain the same number of points.")
X = triangulate(x1, x2,
                projection_matrix(c1["K"], c1["rvec"], c1["tvec"]),
                projection_matrix(c2["K"], c2["rvec"], c2["tvec"]))
with open(a.output, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["index","x","y","z"])
    for i,pnt in enumerate(X): w.writerow([i,*pnt])
print(f"Wrote {a.output} ({len(X)} points)")

#!/usr/bin/env python3
"""Fit a 3D B-spline through triangulated centreline points.

Step 8 of the workflow. Input is a CSV of 3D points (typically produced
by ``triangulate_tube.py``), output is a densely-resampled CSV along the
fitted curve, ready for inspection in FreeCAD.

Example
-------
    fit_curve.py tube_3d.csv -o tube_spline.csv --smoothing 2.0
"""

import argparse
import csv
import numpy as np

from reconstruction.geometry.curves import fit_bspline


def read_points(path):
    """Read 3D points from a CSV; extra columns like 'index' are ignored."""
    with open(path, newline="", encoding="utf-8") as f:
        return np.asarray([[float(r["x"]), float(r["y"]), float(r["z"])]
                           for r in csv.DictReader(f)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="CSV of 3D points along the curve.")
    p.add_argument("-o", "--output", required=True,
                   help="Destination CSV for the sampled spline.")
    p.add_argument("--smoothing", type=float, default=0.0,
                   help="SciPy 's': 0 interpolates, larger smooths.")
    p.add_argument("--samples", type=int, default=200,
                   help="Number of evenly-parameterised samples to output.")
    a = p.parse_args()

    X = read_points(a.input)
    curve, _ = fit_bspline(X, a.smoothing, a.samples)

    with open(a.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "x", "y", "z"])
        for i, pnt in enumerate(curve):
            w.writerow([i, *pnt])
    print(f"Wrote {a.output} ({len(curve)} samples)")


if __name__ == "__main__":
    main()

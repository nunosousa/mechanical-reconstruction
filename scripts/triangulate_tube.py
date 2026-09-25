#!/usr/bin/env python3
"""Triangulate matching tube-centreline clicks from two views into 3D.

Step 7 of the workflow. Both tube CSVs must have the same number of
points, listed in the same order — the i-th row in one file must be the
same physical point as the i-th row in the other.

Example
-------
    triangulate_tube.py cam1.json tube1.csv cam2.json tube2.csv \\
                        -o tube_3d.csv
"""

import argparse
import csv
import numpy as np

from reconstruction.camera.pose import load_camera
from reconstruction.camera.projection import projection_matrix
from reconstruction.reconstruction.triangulation import triangulate


def read_tube(path):
    """Read (u, v) pixel coordinates from a tube CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        return np.asarray([[float(r["u"]), float(r["v"])]
                           for r in csv.DictReader(f)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("camera1", help="JSON camera model for view 1.")
    p.add_argument("tube1", help="Clicked tube points in view 1.")
    p.add_argument("camera2", help="JSON camera model for view 2.")
    p.add_argument("tube2", help="Clicked tube points in view 2.")
    p.add_argument("-o", "--output", required=True,
                   help="Destination CSV of triangulated 3D points.")
    a = p.parse_args()

    c1, c2 = load_camera(a.camera1), load_camera(a.camera2)
    x1, x2 = read_tube(a.tube1), read_tube(a.tube2)
    if len(x1) != len(x2):
        raise ValueError("Tube files must contain the same number of points.")

    X = triangulate(
        x1, x2,
        projection_matrix(c1["K"], c1["rvec"], c1["tvec"]),
        projection_matrix(c2["K"], c2["rvec"], c2["tvec"]),
    )

    with open(a.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "x", "y", "z"])
        for i, pnt in enumerate(X):
            w.writerow([i, *pnt])
    print(f"Wrote {a.output} ({len(X)} points)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Click points along a tube centreline in a photograph, save as CSV.

Step 6 of the workflow. Unlike ``click_landmarks.py`` the number of
points is not fixed — click as many as needed to describe the visible
curve, then press Enter. Run this in two sufficiently different views;
``triangulate_tube.py`` will then lift the pairs into 3D.

Example
-------
    click_tube.py photo.jpg -o results/photo_tube.csv
"""

import argparse
import csv
from pathlib import Path

from reconstruction.image.clicking import click_polyline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", help="Photograph to click the tube on.")
    p.add_argument("-o", "--output",
                   help="Destination CSV; defaults to <image>_tube.csv.")
    a = p.parse_args()

    points = click_polyline(a.image)
    out = a.output or str(
        Path(a.image).with_name(Path(a.image).stem + "_tube.csv"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "u", "v"])
        for i, (u, v) in enumerate(points):
            w.writerow([i, u, v])
    print(f"Wrote {out} ({len(points)} points)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Click each 3D landmark in a photograph and save the pixel coordinates.

Step 3 of the workflow: pair every landmark from ``landmarks.csv`` with
its 2D pixel position in the given image. The prompt in the window tells
you which landmark to click next, and the output CSV keeps the same
row order so downstream tools can pair them by name.

Example
-------
    click_landmarks.py photo.jpg data/casal_k181/measurements/landmarks.csv \\
                       -o results/photo_points.csv
"""

import argparse
from pathlib import Path

from reconstruction.io.csv import read_landmarks_3d, write_points_2d
from reconstruction.image.clicking import click_points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", help="Photograph to click landmarks on.")
    p.add_argument("landmarks_3d", help="CSV of 3D landmarks (name, x, y, z).")
    p.add_argument("-o", "--output",
                   help="Destination CSV; defaults to <image>_points.csv.")
    a = p.parse_args()

    names, _ = read_landmarks_3d(a.landmarks_3d)
    points = click_points(a.image, names)
    out = a.output or str(
        Path(a.image).with_name(Path(a.image).stem + "_points.csv"))
    write_points_2d(out, names, points)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

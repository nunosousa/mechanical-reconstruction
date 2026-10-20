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

Mark circular features for edge fitting with ``circular=true`` in the input
landmarks CSV. For each, click a rough center followed by a point on its
circumference. If fitting fails, the terminal offers a retry, a manual center
click, or the approximate center already selected. Press ``z`` to zoom by
dragging a region without recording that drag as a point; press ``r`` to reset.
The output CSV records the selection method for each point.
"""

import argparse
from pathlib import Path

from reconstruction.io.csv import (
    read_circular_landmarks,
    read_landmarks_3d,
    write_points_2d,
)
from reconstruction.image.clicking import click_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Photograph to click landmarks on.")
    parser.add_argument("landmarks_3d", help="CSV of 3D landmarks (name, x, y, z).")
    parser.add_argument("-o", "--output",
                       help="Destination CSV; defaults to <image>_points.csv.")
    args = parser.parse_args()

    names, _ = read_landmarks_3d(args.landmarks_3d)
    circle_names = read_circular_landmarks(args.landmarks_3d)
    points, methods = click_points(
        args.image, names, circle_names=circle_names)
    out = args.output or str(
        Path(args.image).with_name(Path(args.image).stem + "_points.csv"))
    write_points_2d(out, names, points, methods=methods)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualise reprojection error of a fitted camera against clicked landmarks.

Step 5 of the workflow. For each landmark, projects the known 3D position
through the given camera model and compares it against where it was
clicked. Writes an annotated image (red circles at clicks, green crosses
at projections) and prints per-landmark and RMS pixel errors.

Example
-------
    validate_camera.py photo.jpg landmarks.csv photo_points.csv \\
                       photo_cam.json -o photo_reproj.jpg
"""

import argparse
import cv2
import numpy as np

from reconstruction.io.csv import read_landmarks_3d, read_points_2d
from reconstruction.camera.pose import load_camera
from reconstruction.camera.projection import project_points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", help="Photograph for the visual overlay.")
    p.add_argument("landmarks_3d", help="CSV of 3D landmarks (name, x, y, z).")
    p.add_argument("points_2d", help="CSV of clicked pixels (name, u, v).")
    p.add_argument("camera", help="JSON camera model to validate.")
    p.add_argument("-o", "--output", required=True,
                   help="Destination for the annotated image.")
    a = p.parse_args()

    image = cv2.imread(a.image)
    n3, X = read_landmarks_3d(a.landmarks_3d)
    n2, x = read_points_2d(a.points_2d)
    if n3 != n2:
        raise ValueError("Landmark names/order do not match.")

    cam = load_camera(a.camera)
    xp = project_points(X, cam["K"], cam["rvec"], cam["tvec"], cam["dist"])
    errors = np.linalg.norm(xp - x, axis=1)

    for name, err in zip(n3, errors):
        print(f"{name:24s} {err:7.2f} px")
    print(f"RMS: {np.sqrt(np.mean(errors ** 2)):.2f} px")

    # Red circles = clicked positions; green crosses = model reprojection.
    for actual, predicted in zip(x, xp):
        cv2.circle(image, tuple(np.round(actual).astype(int)),
                   6, (0, 0, 255), 2)
        cv2.drawMarker(image, tuple(np.round(predicted).astype(int)),
                       (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
    cv2.imwrite(a.output, image)
    print(f"Wrote {a.output}")


if __name__ == "__main__":
    main()

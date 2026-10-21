#!/usr/bin/env python3
"""Estimate a camera model from measured 3D landmarks and their 2D clicks.

Step 4 of the workflow. Two modes:

* ``--intrinsics camera.json`` (step 4a): the camera was calibrated with
  ``calibrate_camera.py``; keep its K and distortion and solve only for the
  pose with solvePnP.
* no ``--intrinsics`` (step 4b, historical photographs): jointly fit focal
  length and pose, with zero distortion and a centred principal point.

The estimated camera should be validated with ``validate_camera.py`` before
being used for triangulation.

Example
-------
    fit_camera.py photo.jpg landmarks.csv photo_points.csv -o photo_cam.json
    fit_camera.py photo.jpg landmarks.csv photo_points.csv \\
                  --intrinsics results/camera.json -o photo_cam.json
"""

import argparse

import numpy as np

from reconstruction.io.csv import read_landmarks_3d, read_points_2d
from reconstruction.reconstruction.fitting import fit_camera_unknown_focal
from reconstruction.camera.pose import (
    check_image_size,
    load_camera,
    save_camera,
    solve_pnp,
)
from reconstruction.camera.projection import project_points
from reconstruction.image.preprocessing import image_size, load_image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", help="Photograph the pixel clicks came from.")
    p.add_argument("landmarks_3d", help="CSV of 3D landmarks (name, x, y, z).")
    p.add_argument("points_2d",
                   help="CSV of clicked pixel positions (name, u, v).")
    p.add_argument("-o", "--output", required=True,
                   help="Destination JSON file for the camera model.")
    p.add_argument("--focal-guess", type=float,
                   help="Optional initial focal length in pixels "
                        "(ignored with --intrinsics).")
    p.add_argument("--intrinsics",
                   help="Calibrated camera JSON; solve only for pose (PnP).")
    a = p.parse_args()

    n3, X = read_landmarks_3d(a.landmarks_3d)
    n2, x = read_points_2d(a.points_2d)
    # The two files must list the same landmarks in the same order — the
    # fit pairs them by index, not by name.
    if n3 != n2:
        raise ValueError("3D and 2D landmark names/order do not match.")

    size = image_size(load_image(a.image))

    if a.intrinsics:
        camera = load_camera(a.intrinsics)
        check_image_size(camera, size)
        K, dist = camera["K"], camera["dist"]
        rvec, tvec = solve_pnp(X, x, K, dist)
        errors = np.linalg.norm(project_points(X, K, rvec, tvec, dist) - x, axis=1)
        rms = float(np.sqrt(np.mean(errors ** 2)))
    else:
        K, dist, rvec, tvec, rms, result = fit_camera_unknown_focal(
            X, x, size, a.focal_guess)
        print(f"success: {result.success}")
        print(f"estimated focal length: {K[0, 0]:.2f} px")
    save_camera(a.output, K, dist, rvec, tvec, size, rms)

    print(f"RMS reprojection error: {rms:.3f} px")
    print(f"Wrote {a.output}")


if __name__ == "__main__":
    main()

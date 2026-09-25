#!/usr/bin/env python3
"""Estimate a camera model from measured 3D landmarks and their 2D clicks.

Step 4b of the workflow: use this for historical photographs, where we
have no checkerboard calibration. Jointly fits focal length and camera
pose so that the 3D landmarks reproject onto the clicked pixel
positions. The estimated camera should be validated with
``validate_camera.py`` before being used for triangulation.

Example
-------
    fit_camera.py photo.jpg landmarks.csv photo_points.csv -o photo_cam.json
"""

import argparse
import cv2

from reconstruction.io.csv import read_landmarks_3d, read_points_2d
from reconstruction.reconstruction.fitting import fit_camera_unknown_focal
from reconstruction.camera.pose import save_camera


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", help="Photograph the pixel clicks came from.")
    p.add_argument("landmarks_3d", help="CSV of 3D landmarks (name, x, y, z).")
    p.add_argument("points_2d",
                   help="CSV of clicked pixel positions (name, u, v).")
    p.add_argument("-o", "--output", required=True,
                   help="Destination JSON file for the camera model.")
    p.add_argument("--focal-guess", type=float,
                   help="Optional initial focal length in pixels.")
    a = p.parse_args()

    n3, X = read_landmarks_3d(a.landmarks_3d)
    n2, x = read_points_2d(a.points_2d)
    # The two files must list the same landmarks in the same order — the
    # fit pairs them by index, not by name.
    if n3 != n2:
        raise ValueError("3D and 2D landmark names/order do not match.")

    image = cv2.imread(a.image)
    if image is None:
        raise FileNotFoundError(a.image)
    h, w = image.shape[:2]

    K, dist, rvec, tvec, rms, result = fit_camera_unknown_focal(
        X, x, (w, h), a.focal_guess)
    save_camera(a.output, K, dist, rvec, tvec, (w, h), rms)

    print(f"success: {result.success}")
    print(f"RMS reprojection error: {rms:.3f} px")
    print(f"estimated focal length: {K[0, 0]:.2f} px")
    print(f"Wrote {a.output}")


if __name__ == "__main__":
    main()

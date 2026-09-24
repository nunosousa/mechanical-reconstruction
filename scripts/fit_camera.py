#!/usr/bin/env python3
import argparse
import cv2
from reconstruction.io.csv import read_landmarks_3d, read_points_2d
from reconstruction.reconstruction.fitting import fit_camera_unknown_focal
from reconstruction.camera.pose import save_camera

p = argparse.ArgumentParser()
p.add_argument("image"); p.add_argument("landmarks_3d"); p.add_argument("points_2d")
p.add_argument("-o", "--output", required=True); p.add_argument("--focal-guess", type=float)
a = p.parse_args()

n3, X = read_landmarks_3d(a.landmarks_3d)
n2, x = read_points_2d(a.points_2d)
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
print(f"estimated focal length: {K[0,0]:.2f} px")
print(f"Wrote {a.output}")

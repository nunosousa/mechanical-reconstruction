#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
from reconstruction.io.csv import read_landmarks_3d, read_points_2d
from reconstruction.camera.pose import load_camera
from reconstruction.camera.projection import project_points

p = argparse.ArgumentParser()
p.add_argument("image"); p.add_argument("landmarks_3d"); p.add_argument("points_2d")
p.add_argument("camera"); p.add_argument("-o", "--output", required=True)
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
print(f"RMS: {np.sqrt(np.mean(errors**2)):.2f} px")

for actual, predicted in zip(x, xp):
    cv2.circle(image, tuple(np.round(actual).astype(int)), 6, (0,0,255), 2)
    cv2.drawMarker(image, tuple(np.round(predicted).astype(int)),
                   (0,255,0), cv2.MARKER_CROSS, 14, 2)
cv2.imwrite(a.output, image)
print(f"Wrote {a.output}")

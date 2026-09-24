#!/usr/bin/env python3
import argparse
from pathlib import Path
from reconstruction.io.csv import read_landmarks_3d, write_points_2d
from reconstruction.image.clicking import click_points

p = argparse.ArgumentParser()
p.add_argument("image"); p.add_argument("landmarks_3d")
p.add_argument("-o", "--output")
a = p.parse_args()

names, _ = read_landmarks_3d(a.landmarks_3d)
points = click_points(a.image, names)
out = a.output or str(Path(a.image).with_name(Path(a.image).stem + "_points.csv"))
write_points_2d(out, names, points)
print(f"Wrote {out}")

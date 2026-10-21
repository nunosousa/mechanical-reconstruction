#!/usr/bin/env python3
"""Camera intrinsic calibration from checkerboard photographs.

Step 4a of the workflow (used when we control the camera). Feed one or
more shell globs pointing at checkerboard photos and write a JSON camera
file containing K, distortion, and image size. The saved rvec / tvec are
zeroed because the checkerboard pose is meaningless in the reconstruction
coordinate system — pose comes later via ``fit_camera.py --intrinsics``.

Example
-------
    calibrate_camera.py "calib/*.jpg" -o results/camera.json
"""

import argparse
import glob

from reconstruction.camera.calibration import calibrate_checkerboard
from reconstruction.camera.pose import save_camera


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", nargs="+",
                   help="Glob(s) matching checkerboard photos.")
    p.add_argument("-o", "--output", required=True,
                   help="Destination JSON file for the camera model.")
    p.add_argument("--cols", type=int, default=9,
                   help="Number of inner corners per row of the board.")
    p.add_argument("--rows", type=int, default=6,
                   help="Number of inner corners per column of the board.")
    p.add_argument("--square-size", type=float, default=25.0,
                   help="Physical size of one board square (mm by convention).")
    p.add_argument("--debug-dir",
                   help="Write annotated copies of all images for corner verification.")
    a = p.parse_args()

    # Accept both real globs and plain filenames — glob() returns [] for
    # a literal path, in which case we fall back to using it verbatim.
    paths = []
    for pattern in a.images:
        paths.extend(glob.glob(pattern) or [pattern])

    rms, K, dist, rvecs, tvecs, image_size = calibrate_checkerboard(
        paths, (a.cols, a.rows), a.square_size, debug_dir=a.debug_dir)

    # Only intrinsics are meaningful here; rvec/tvec placeholders are
    # replaced by ``fit_camera.py --intrinsics`` (solvePnP) against real
    # scene landmarks.
    save_camera(a.output, K, dist, [0, 0, 0], [0, 0, 0], image_size, rms)

    print(f"usable images: {len(rvecs)}")
    print(f"calibration RMS: {rms:.4f} px")
    print(K)
    print(dist.ravel())
    print(f"Wrote {a.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse, glob, cv2
from reconstruction.camera.calibration import calibrate_checkerboard
from reconstruction.camera.pose import save_camera

p=argparse.ArgumentParser()
p.add_argument("images",nargs="+"); p.add_argument("-o","--output",required=True)
p.add_argument("--cols",type=int,default=9); p.add_argument("--rows",type=int,default=6)
p.add_argument("--square-size",type=float,default=25.0)
a=p.parse_args()

paths=[]
for pattern in a.images:
    paths.extend(glob.glob(pattern) or [pattern])
rms,K,dist,rvecs,tvecs=calibrate_checkerboard(
    paths,(a.cols,a.rows),a.square_size)
image=cv2.imread(paths[0])
save_camera(a.output,K,dist,[0,0,0],[0,0,0],
            (image.shape[1],image.shape[0]),rms)
print(f"usable images: {len(rvecs)}")
print(f"calibration RMS: {rms:.4f} px")
print(K)
print(dist.ravel())
print(f"Wrote {a.output}")

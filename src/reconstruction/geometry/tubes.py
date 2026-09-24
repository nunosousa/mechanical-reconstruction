import numpy as np

def tube_mesh_placeholder(centerline, radius, sides=16):
    centerline = np.asarray(centerline, float)
    if radius <= 0:
        raise ValueError("radius must be positive")
    return centerline, float(radius), int(sides)

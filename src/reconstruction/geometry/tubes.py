"""Placeholder for tube-shaped mesh generation.

The long-term intent (see README) is to sweep a circular cross-section
along a fitted centreline to produce a solid tube for FreeCAD. That's not
implemented yet — this module currently just validates its inputs and
echoes them back, so downstream code can be wired up first.
"""

import numpy as np


def tube_mesh_placeholder(centerline, radius, sides=16):
    """Placeholder: validate inputs and return them, no meshing yet."""
    centerline = np.asarray(centerline, float)
    if radius <= 0:
        raise ValueError("radius must be positive")
    return centerline, float(radius), int(sides)

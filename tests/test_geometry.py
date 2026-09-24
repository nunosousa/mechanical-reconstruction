import numpy as np
from reconstruction.geometry.points import pairwise_distances

def test_pairwise_distance():
    d=pairwise_distances([[0,0,0],[3,4,0]])
    assert d[0,1]==5

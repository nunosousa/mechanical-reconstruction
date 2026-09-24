import numpy as np
from reconstruction.camera.projection import camera_matrix, projection_matrix
from reconstruction.reconstruction.triangulation import triangulate

def test_simple_triangulation():
    K=camera_matrix(1000,(2000,1000))
    P1=projection_matrix(K,[0,0,0],[0,0,0])
    P2=projection_matrix(K,[0,0,0],[100,0,0])
    X=triangulate([[1000,500]],[[900,500]],P1,P2)
    np.testing.assert_allclose(X,[[0,0,1000]],atol=1e-5)

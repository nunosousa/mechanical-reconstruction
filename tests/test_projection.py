import numpy as np
from reconstruction.camera.projection import camera_matrix, project_points

def test_projection_centre():
    K=camera_matrix(1000,(2000,1000))
    uv=project_points([[0,0,1000]],K,[0,0,0],[0,0,0])
    np.testing.assert_allclose(uv,[[1000,500]])

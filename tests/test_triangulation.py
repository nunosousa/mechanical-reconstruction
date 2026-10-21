import numpy as np
from reconstruction.camera.projection import camera_matrix, projection_matrix
from reconstruction.reconstruction.triangulation import triangulate

def test_simple_triangulation():
    K=camera_matrix(1000,(2000,1000))
    P1=projection_matrix(K,[0,0,0],[0,0,0])
    P2=projection_matrix(K,[0,0,0],[-100,0,0])
    X=triangulate([[1000,500]],[[900,500]],P1,P2)
    np.testing.assert_allclose(X,[[0,0,1000]],atol=1e-5)


def test_triangulation_with_lens_distortion():
    from reconstruction.camera.projection import project_points, undistort_pixels

    K = camera_matrix(1000, (2000, 1000))
    dist = np.array([0.1, -0.3, 0.001, -0.002, 0.5])
    X = np.array([[150., -120., 900.], [-200., 80., 1100.]])
    r2, t2 = [0, 0.2, 0], [-300, 0, 50]
    x1 = project_points(X, K, [0, 0, 0], [0, 0, 0], dist)
    x2 = project_points(X, K, r2, t2, dist)

    Xr = triangulate(undistort_pixels(x1, K, dist), undistort_pixels(x2, K, dist),
                     projection_matrix(K, [0, 0, 0], [0, 0, 0]),
                     projection_matrix(K, r2, t2))
    np.testing.assert_allclose(Xr, X, atol=1e-3)

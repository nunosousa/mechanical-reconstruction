import cv2

def canny_edges(image, low=50, high=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, low, high)

def distance_transform_from_edges(edges):
    mask = (edges == 0).astype("uint8")
    return cv2.distanceTransform(mask, cv2.DIST_L2, 3)

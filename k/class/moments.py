import numpy as np
import cv2 as cv
import os
img = cv.imread('moon.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
ret,thresh = cv.threshold(img,127,255,0)
contours,hierarchy = cv.findContours(thresh, 1, 3)
cnt = contours[0]
M = cv.moments(cnt)
print( M )

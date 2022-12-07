import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#img = cv.imread('images/clean.png',0)
img = cv.imread('images/neu.png',0)
edges = cv.Canny(img, 150, 300)
edges, thr = cv.threshold(edges, 200, 255, cv.THRESH_BINARY)

plt.subplot(121),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
import numpy as np
import cv2
import math 
from copy import *
from sympy import resultant
from tqdm import tqdm 
from matplotlib import pyplot as plt

# KMeans segmenter code Refs - https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python, https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html
def kmeans_segmenter(img, k, max_iter=50):
    color_vals = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    color_vals = np.float32(color_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)
    ret, labels, (centers) = cv2.kmeans(color_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    resultant_img = centers[labels.flatten()]
    resultant_img = np.reshape(resultant_img, (img.shape))
    resultant_img = cv2.cvtColor(resultant_img, cv2.COLOR_BGR2RGB)
    cv2.imshow("K-Means Image", resultant_img)
    cv2.imwrite("kmeans_bigtree.png", resultant_img)
    return resultant_img

img = cv2.imread("./BigTree.jpeg")
cv2.imshow("Original Image",img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = kmeans_segmenter(img, 85)
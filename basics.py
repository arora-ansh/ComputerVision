import numpy as np
import cv2
import math 
from copy import *
from tqdm import tqdm 
from matplotlib import pyplot as plt

# Image import code 
img = cv2.imread("./p1.jpg")
img_gs = cv2.imread("./p1.jpg",0)
cv2.imshow("Coloured Original Image",img)
cv2.imshow("Grayscale Original Image",img_gs)

img_bi = np.zeros((len(img_gs),len(img_gs[0])))
for i in range(len(img_gs)):
    for j in range(len(img_gs[0])):
        if img_gs[i][j]<128:
            img_bi[i][j] = 0
        else:
            img_bi[i][j] = 255

img_bi = np.array(img_bi,dtype=np.uint8)
cv2.imshow("Binary Image",img_bi)
cv2.waitKey(0)
cv2.destroyAllWindows()

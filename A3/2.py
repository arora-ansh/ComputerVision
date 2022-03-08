import numpy as np
import cv2
import math 
from copy import *
from tqdm import tqdm 
from matplotlib import pyplot as plt
import os
import pandas as pd
from sklearn.cluster import DBSCAN
import time

img = cv2.imread('./data/0002.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
m = img.shape[0]
n = img.shape[1]

# Create data matrix of the form [x, y, r, g, b]
X = []
for i in range(m):
    for j in range(n):
        X.append([i, j, img[i, j, 0], img[i, j, 1], img[i, j, 2]])

X = np.array(X)

# Normalize the data by dividing each row by [m,n,255,255,255]
nor_X = X / np.array([m, n, 255, 255, 255])

# We will now create a DBSCAN model using the data matrix X and will time it
# to see how long it takes to run.
for x in tqdm(range(1,21)):

    t = time.time()
    db = DBSCAN(eps=0.0025*x, min_samples=5, metric='euclidean').fit(nor_X)
    print("Time taken:", time.time() - t)
    X_labels = db.labels_

    # print(X_labels)

    # We will allot each label a unique color. We will do this from X 
    label_color_dict = {}
    for i in range(len(X_labels)):
        if X_labels[i] not in label_color_dict:
            label_color_dict[X_labels[i]] = [X[i, 2], X[i, 3], X[i, 4]]

    print(len(label_color_dict.keys()))

    # We will now create a new image with the same dimensions as the original image
    # and will color each pixel according to its cluster label.
    new_img = np.zeros((m, n, 3))
    for i in range(len(X)):
        # We will assign each pixel a unique color from the label_color_dict
        new_img[X[i][0]][X[i][1]][0] = label_color_dict[X_labels[i]][0]
        new_img[X[i][0]][X[i][1]][1] = label_color_dict[X_labels[i]][1]
        new_img[X[i][0]][X[i][1]][2] = label_color_dict[X_labels[i]][2]

    # We will now save the new image as a png file and display it.
    # First convert the image into uint8 format
    new_img = new_img.astype(np.uint8)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./output/Q2/'+str(x)+'.jpg', new_img)

# cv2.imshow('image', new_img)
# cv2.waitKey(0)

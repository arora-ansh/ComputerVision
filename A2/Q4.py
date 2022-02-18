import numpy as np
import cv2
import math 
from copy import *
from tqdm import tqdm 
from matplotlib import pyplot as plt
import os

# Read all the images in the folder data/dd

def load_images_from_folder(folder):
    images = []
    print("Loading images from folder: ",folder)
    for filename in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename),0) #Inputting images in grayscale
        if img is not None:
            images.append(img)
    return images

images = load_images_from_folder("./data/dd")

# We will find 10 corners using Shi-Thomsi https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html in each of the images and store them in a list

total_images_corner_vector = []
total_lbp_features = []

for img in tqdm(images):
    img = cv2.resize(img,(128,64))
    corners = cv2.goodFeaturesToTrack(img,10,0.01,10)
    corners = np.int0(corners)
    total_images_corner_vector.append(corners)

    #We will find the LBP for patch drawn around corners

# print(total_images_corner_vector[0][0][0][1])

query_img = cv2.imread("./data/15_9_s.jpg",0)



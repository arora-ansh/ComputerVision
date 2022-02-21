import numpy as np
import cv2
import math 
from copy import *
from tqdm import tqdm 
from matplotlib import pyplot as plt
import os
from pathlib import Path

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
folder = "./data/dd"
# We will find 10 corners using Shi-Thomsi https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html in each of the images and store them in a list

total_images_corner_vector = []
total_lbp_features = []

for img in tqdm(images):
    img = cv2.resize(img,(128,64))
    corners = cv2.goodFeaturesToTrack(img,5,0.01,10)
    corners = np.int0(corners)
    total_images_corner_vector.append(corners)

    #We will find the LBP for patch drawn around corners
    #For this we will consider a 3*3 8-pixel patch around the corner pixel in question, calculate the LBP for all of the 9 pixels and take their median
    lbp_features = []
    for corner in corners:
        x,y = corner[0][0],corner[0][1]
        if x>1 and y>1 and x<126 and y<62:
            patch_lbp = np.zeros((3,3))
            for i in range(3):
                for j in range(3):
                    cur_centre_x, cur_centre_y = x-1+i, y-1+j
                    cur_patch = img[cur_centre_y-1:cur_centre_y+2,cur_centre_x-1:cur_centre_x+2]
                    # print("lol",cur_patch,cur_centre_x,cur_centre_y)
                    cur_patch = cur_patch.flatten()
                    for k in range(len(cur_patch)):
                        if cur_patch[k]>img[cur_centre_y,cur_centre_x]:
                            cur_patch[k] = 1
                        else:
                            cur_patch[k] = 0
                    multiplier = [128,64,32,1,0,16,2,4,8]
                    cur_patch = cur_patch*multiplier
                    patch_lbp[i,j] = np.sum(cur_patch)
            lbp_features.append(np.median(patch_lbp))
        elif x>0 and y<0 and x<127 and y<63:
            cur_centre_x, cur_centre_y = x,y
            cur_patch = img[cur_centre_y-1:cur_centre_y+2,cur_centre_x-1:cur_centre_x+2]
            cur_patch = cur_patch.flatten()
            for k in range(len(cur_patch)):
                if cur_patch[k]>img[cur_centre_y,cur_centre_x]:
                    cur_patch[k] = 1
                else:
                    cur_patch[k] = 0
            multiplier = [128,64,32,1,0,16,2,4,8]
            cur_patch = cur_patch*multiplier
            lbp_features.append(np.sum(cur_patch))
        else:
            lbp_features.append(0)

    total_lbp_features.append(lbp_features)
                    
            

# print(total_images_corner_vector[0][0][0][1])

query_img = cv2.imread("./data/15_19_s.jpg",0)
query_img = cv2.resize(query_img,(128,64))
query_corners = cv2.goodFeaturesToTrack(query_img,15,0.01,5)
query_corners = np.int0(query_corners)

query_lbp_features = []

for corner in query_corners:
    x,y = corner[0][0],corner[0][1]
    if x>1 and y>1 and x<126 and y<62:
        patch_lbp = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                cur_centre_x, cur_centre_y = x-1+i, y-1+j
                cur_patch = query_img[cur_centre_y-1:cur_centre_y+2,cur_centre_x-1:cur_centre_x+2]
                cur_patch = cur_patch.flatten()
                for k in range(len(cur_patch)):
                    if cur_patch[k]>query_img[cur_centre_y,cur_centre_x]:
                        cur_patch[k] = 1
                    else:
                        cur_patch[k] = 0
                multiplier = [128,64,32,1,0,16,2,4,8]
                cur_patch = cur_patch*multiplier
                patch_lbp[i,j] = np.sum(cur_patch)
        query_lbp_features.append(np.median(patch_lbp))
    elif x>0 and y<0 and x<127 and y<63:
        cur_centre_x, cur_centre_y = x,y
        cur_patch = query_img[cur_centre_y-1:cur_centre_y+2,cur_centre_x-1:cur_centre_x+2]
        cur_patch = cur_patch.flatten()
        for k in range(len(cur_patch)):
            if cur_patch[k]>query_img[cur_centre_y,cur_centre_x]:
                cur_patch[k] = 1
            else:
                cur_patch[k] = 0
        print(cur_patch)
        multiplier = [128,64,32,1,0,16,2,4,8]
        cur_patch = cur_patch*multiplier
        query_lbp_features.append(np.sum(cur_patch))
    else:
        query_lbp_features.append(0)

k = 5

errors = []

for lbp in total_lbp_features:
    cur_error = 0
    for j in range(len(lbp)):
        for i in range(len(query_lbp_features)):
            cur_error += math.sqrt((lbp[j]-query_lbp_features[i])**2)
    errors.append(cur_error)

#Now just sort and return the least k error images 
error_vector = []
for i in range(len(errors)):
    error_vector.append((errors[i],i))

#We will now sort the error vector
error_vector.sort()
print(error_vector)

error_vector = error_vector[:k]

Path("./output/Q4").mkdir(parents=True, exist_ok=True)

count = 0
for filename in tqdm(os.listdir(folder)):
    img = cv2.imread(os.path.join(folder,filename)) #Inputting images in grayscale
    if img is not None:
        for x in error_vector:
            if x[1] == count:
                cv2.imwrite("./output/Q4/"+filename,img)
    count+=1


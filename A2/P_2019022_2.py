import numpy as np
import cv2
import math 
from copy import *
from tqdm import tqdm 
from matplotlib import pyplot as plt
import os
from fcmeans import FCM
from pathlib import Path

def modified_LBP(patch):
    """
    Takes in the Patch Matrix and returns the modified LBP's mean and standard deviation to form features
    """
    # Calculate the modified LBP for each of the pixels in the patch and store their values in the lbp matrix 
    # First create a zero padded matrix of size (patch_size+2)x(patch_size+2)
    
    zero_padded_patch = np.zeros((len(patch)+2,len(patch[0])+2))
    # Then copy the patch matrix into the zero padded matrix
    for i in range(len(patch)):
        for j in range(len(patch[0])):
            zero_padded_patch[i+1][j+1] = patch[i][j]

    lbp = np.zeros((len(patch),len(patch[0])))
    for i in range(1,len(patch)+1):
        for j in range(1,len(patch[0])+1):
            lbp_val = 0
            C = zero_padded_patch[i][j]
            for k in range(8):
                try:
                    if k==0:
                        N = zero_padded_patch[i][j-1]
                        lbp_val += round(min(N,C)/max(N,C))*1
                    elif k==1:
                        N = zero_padded_patch[i+1][j-1]
                        lbp_val += round(min(N,C)/max(N,C))*2
                    elif k==2:
                        N = zero_padded_patch[i+1][j]
                        lbp_val += round(min(N,C)/max(N,C))*4
                    elif k==3:
                        N = zero_padded_patch[i+1][j+1]
                        lbp_val += round(min(N,C)/max(N,C))*8
                    elif k==4:
                        N = zero_padded_patch[i][j+1]
                        lbp_val += round(min(N,C)/max(N,C))*16
                    elif k==5:
                        N = zero_padded_patch[i-1][j+1]
                        lbp_val += round(min(N,C)/max(N,C))*32
                    elif k==6:
                        N = zero_padded_patch[i-1][j]
                        lbp_val += round(min(N,C)/max(N,C))*64
                    elif k==7:
                        N = zero_padded_patch[i-1][j-1]
                        lbp_val += round(min(N,C)/max(N,C))*128
                except:
                    lbp_val += 0 #Handles the NaN image pixel value edge case observed in 2 images for 3 pixels
            # print(lbp_val)
            lbp[i-1][j-1] = lbp_val 
    
    #The lbp matrix now holds the lbp_values, we will now find the mean and standard deviation of the lbp values
    mean = np.mean(lbp)
    std = np.std(lbp)
    return mean,std
            

def feature_extract(img):
    """
    First we will divide the image into 16 patches, 4 patches and single patch
    Then we will calculate the mean and standard deviation of the modified LBP for each patch
    Then we will concatenate the mean and standard deviation values into a single feature vector
    Finally we will return the feature vector, which will have 42 features
    """
    feature_vector = []
    # Divide the image into 16 patches
    for i in range(0,len(img),int(len(img)/4)):
        for j in range(0,len(img[0]),int(len(img[0])/4)):
            # Take the patch
            patch = img[i:i+int(len(img[0])/4),j:j+int(len(img[0])/4)]
            # Calculate the mean and standard deviation of the modified LBP
            mean,std = modified_LBP(patch)
            # Append the mean and standard deviation to the feature vector
            feature_vector.append(mean)
            feature_vector.append(std)
    
    # Divide the image into 4 patches
    for i in range(0,len(img),int(len(img)/2)):
        for j in range(0,len(img[0]),int(len(img[0])/2)):
            # Take the patch
            patch = img[i:i+int(len(img[0])/2),j:j+int(len(img[0])/2)]
            # Calculate the mean and standard deviation of the modified LBP
            mean,std = modified_LBP(patch)
            # Append the mean and standard deviation to the feature vector
            feature_vector.append(mean)
            feature_vector.append(std)
    
    # Take the single patch
    patch = img
    # Calculate the mean and standard deviation of the modified LBP
    mean,std = modified_LBP(patch)
    # Append the mean and standard deviation to the feature vector
    feature_vector.append(mean)
    feature_vector.append(std)

    return feature_vector

# Read all the images in the folder data/dd

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0) #Inputting images in grayscale
        if img is not None:
            images.append(img)
    return images

images = load_images_from_folder("./data/dd")

total_feature_vector = []

#Generate the feature vector for each of the images
for i in tqdm(range(len(images))):
    # Reduce the image into dimensions of multiples of 4
    images[i] = cv2.resize(images[i],(128,64))
    feature_vector = feature_extract(images[i])
    total_feature_vector.append(feature_vector)

k = int(input("Enter value for k:")) #Can change value of k from here

total_feature_vector = np.array(total_feature_vector)
print(total_feature_vector.shape)

fcm = FCM(n_clusters=k)
fcm.fit(total_feature_vector)
fcm_labels = fcm.predict(total_feature_vector)

fcm_centers = np.array(fcm.centers)

# print(fcm_centers.shape)
print(fcm_labels)

#Creating the folders for the image clusters
Path("./output/Q2").mkdir(parents=True, exist_ok=True)
for i in range(k):
    Path("./output/Q2/cluster_"+str(i+1)).mkdir(parents=True, exist_ok=True)

count = 0
for filename in os.listdir("./data/dd"):
    img = cv2.imread(os.path.join("./data/dd",filename)) #Inputting images in grayscale
    if img is not None:
        cluster = fcm_labels[count]
        cv2.imwrite("./output/Q2/cluster_"+str(cluster+1)+"/"+filename,img)
        count += 1
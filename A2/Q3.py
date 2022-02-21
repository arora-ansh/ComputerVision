import numpy as np
import cv2
import math 
from copy import *
from tqdm import tqdm 
from matplotlib import pyplot as plt
import os
from skimage.feature import hog
from sklearn.cluster import KMeans

# Read all the images in the folder data/dd

def load_images_from_folder(folder):
    images = []
    print("Loading images from folder: ",folder)
    for filename in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename)) #Inputting images in RGB
        if img is not None:
            images.append(img)
    return images

images = load_images_from_folder("./data/dd")

# https://scikit-image.org/docs/stable/api/skimage.feature.html?highlight=hog#skimage.feature.hog
# We will resize all the images to 128*64 and then calculate the HOG feature vector for each of the images

total_feature_vector = []

print("Calculating HOG feature vector for each of the images")
for img in tqdm(images):
    img = cv2.resize(img,(128,64))
    feature_vector = hog(img, orientations=8, pixels_per_cell=(8,16), cells_per_block=(1, 1), visualize=False)
    total_feature_vector.append(feature_vector)

# print(len(total_feature_vector), len(total_feature_vector[0]))
# print(feature_vector)
# plt.imshow(hog_image, cmap=plt.cm.gray)
# plt.show()

# We will now calculate the top k Bag of Visual Words from the feature vectors
visual_words = []
for i in range(len(total_feature_vector)):
    for j in range(0,len(total_feature_vector[i]),8):
        visual_words.append(total_feature_vector[i][j:j+8])
print("Length of visual words: ",len(visual_words))
# Each patch's HoG has now been stored into a list of visual words
# We will now find the k top words using K-Means clustering, which form the 
k = 100
kmeans = KMeans(n_clusters=k, random_state=0).fit(visual_words)

visual_words_dictionary = kmeans.cluster_centers_
print("Number of iterations: ",kmeans.n_iter_)
print(visual_words_dictionary)
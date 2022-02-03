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
    cv2.imwrite("kmeans_leaf.png", resultant_img)
    return resultant_img

#We first take in the image in color format, with the image being reduced to 85 using kmeans function defined above, Run this if kmeans_leaf.png doesnt exist
# img = cv2.imread("./leaf.png")
# cv2.imshow("Original Image",img)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = kmeans_segmenter(img, 85)

#If above code has been run and kmeans_leaf.png exists, then run this
img = cv2.imread("./kmeans_leaf.png")
cv2.imshow("85 color Image",img)

#We now need to calculate the color distance metric for each pixel in the image. For this we simply find the distance between two color values used in the image, c_l and c_j.
m = len(img)
n = len(img[0])
#We first lay out all the pixel values in a single set
img_pixel_vals = set()
for i in range(m):
    for j in range(n):
        img_pixel_vals.add(tuple(img[i][j]))

print(len(img_pixel_vals)) #This should be 85

color_list = []
color_list_dict = {}
for i in img_pixel_vals:
    color_list.append(i)
    color_list[-1] = np.array(color_list[-1])/255
    color_list_dict[i] = len(color_list)-1

#We will find the frequency of each of the 85 colours in the image.
color_freq = np.zeros(len(img_pixel_vals))
for i in range(m):
    for j in range(n):
        color_freq[color_list_dict[tuple(img[i][j])]] += 1
#Normalizing the color frequency
color_freq = color_freq/np.sum(color_freq)
#Plotting the color frequency for each of the colors of the image
plt.bar(range(len(img_pixel_vals)), color_freq)
plt.show()

#We now create a dictionary to store the color distance metric for each color value.
#We take the distance between colors to be Euclidean distance.
color_dict = {}
for i in tqdm(range(len(color_list))):
    color_dict[(i,i)] = 0
    #We now iterate through the image, and for each pixel, we find the distance between the pixel's color value and the color values in the color_list.
    for j in range(len(color_list)):
        if i != j and (j,i) not in color_dict.keys():
            color_dict[(i,j)] = math.sqrt(sum((color_list[i]-color_list[j])**2))

#We now have D(ci,cj) for each color value ci in the image.
# print(color_dict)

#Now aplying the formula to generate saliency values and storing them in an image
saliency_img = np.zeros((m,n))
for i in tqdm(range(m)):
    for j in range(n):
        cur_pix_color_idx = color_list_dict[tuple(img[i][j])]
        for keys in color_dict.keys():
            if keys[0] == cur_pix_color_idx:
                saliency_img[i][j] += color_dict[keys]*color_freq[keys[1]]
            elif keys[1] == cur_pix_color_idx:
                saliency_img[i][j] += color_dict[keys]*color_freq[keys[0]]

cv2.imshow("Saliency Image Eqn3", 1-saliency_img)
cv2.imwrite("saliency_leaf_eqn3.png", 255*saliency_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np
import cv2
import math 
from copy import *
from tqdm import tqdm 
from matplotlib import pyplot as plt

#We first take in the image as a grayscale (since otsu can only be applied on grayscale images)¡
img_gs = cv2.imread("./leaf.png",0)
cv2.imshow("Grayscale Original Image",img_gs) 

#We find the lowest and the highest values of the image's pixel intensities.
low_intensity = np.amin(img_gs)
high_intensity = np.amax(img_gs)

print(low_intensity, high_intensity)

#We will now calculate the histogram for the given gray scale image.
hist = cv2.calcHist([img_gs],[0],None,[256],[0,256])

#We will now plot the histogram.
plt.plot(hist)
plt.show()

#We create an array to store the TSS values obtained for each threshold, which we will then use to find the optimal threshold, and also store as a CSV.
TSS = []

min_tss = np.inf
min_threshold = -1
for threshold in range(low_intensity,high_intensity):
    # The threshold divides the image into two classes - less than equal to the threshold and greater than the threshold.
    #We will first find the means in each class.
    class1 = hist[:threshold+1]
    class2 = hist[threshold+1:]

    # Function to calculate mean of each of the two classes.
    def mean_calc(X,add=0):
        mean = 0
        count = 0
        for i in range(len(X)):
            mean = mean + X[i]*(i+add)
            count = count + X[i]
        mean = mean/count
        return int(mean)

    mean1 = mean_calc(class1)
    mean2 = mean_calc(class2,threshold)

    # Function to calculate the total sum of squares for each of the two classes.
    def sum_of_sqaures(X,mean,add=0):
        sum = 0
        for i in range(len(X)):
            sum = sum + X[i]*(((i+add)-mean)**2)
        return sum[0]

    ss1 = sum_of_sqaures(class1,mean1)
    ss2 = sum_of_sqaures(class2,mean2,threshold)
    tss = ss1+ss2 # Total sum of squares.
    # print(threshold, mean1,mean2,ss1,ss2)
    TSS.append(tss)
    if tss<min_tss:
        min_tss = tss
        min_threshold = threshold

print("Minimum Threshold:",min_threshold)

#We now create a binary mask using the optimal threshold.
img_bi = np.zeros((len(img_gs),len(img_gs[0])))
for i in range(len(img_gs)):
    for j in range(len(img_gs[0])):
        if img_gs[i][j]<min_threshold:
            img_bi[i][j] = 0
        else:
            img_bi[i][j] = 255

#Showing the binary image.
img_bi = np.array(img_bi,dtype=np.uint8)
cv2.imshow("Binary Image",img_bi)

#Saving the binary image.
cv2.imwrite("leaf_binmask.png",img_bi)

#Saving the TSS values obtained in a CSV file.
TSS_CSV = []
for i in range(len(TSS)):
    TSS_CSV.append([int(i),TSS[i]])
np.savetxt("leaf_tss.csv",TSS_CSV,delimiter=",")

cv2.waitKey(0)
cv2.destroyAllWindows()
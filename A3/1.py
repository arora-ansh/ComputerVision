import numpy as np
import cv2
import math 
from copy import *
from tqdm import tqdm 
from matplotlib import pyplot as plt
import os
import pandas as pd
import time

img = cv2.imread('./data/0002.jpg')
m = img.shape[0]
n = img.shape[1]
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

# Modify SLIC parameters from here
region_size = 10
ruler = 5.0

# We will use img_lab to calculate SLIC superpixels https://docs.opencv.org/3.4/d3/da9/classcv_1_1ximgproc_1_1SuperpixelSLIC.html 
t = time.time()
slic = cv2.ximgproc.createSuperpixelSLIC(img_lab, region_size = region_size, ruler = ruler)
slic.iterate(1000)
print("Time taken:", time.time() - t)
super_pixels_labels = slic.getLabels()
num_super_pixels = slic.getNumberOfSuperpixels()

print("Number of superpixels: {}".format(num_super_pixels))

# We now have to find average location color for each superpixel. We will use a dictionary for this.

super_pixel_dict = {} # [superpixel_id] = [number_of_pixels, mean_x, mean_y, mean_r, mean_g, mean_b]

for i in range(m):
    for j in range(n):
        sp_val = super_pixels_labels[i][j]
        if sp_val not in super_pixel_dict.keys():
            super_pixel_dict[sp_val] = []
            super_pixel_dict[sp_val].append(float(1))
            super_pixel_dict[sp_val].append(float(i))
            super_pixel_dict[sp_val].append(float(j))
            super_pixel_dict[sp_val].append(float(img_rgb[i][j][0]))
            super_pixel_dict[sp_val].append(float(img_rgb[i][j][1]))
            super_pixel_dict[sp_val].append(float(img_rgb[i][j][2]))
        else:
            super_pixel_dict[sp_val][0] += 1
            super_pixel_dict[sp_val][1] += i
            super_pixel_dict[sp_val][2] += j
            super_pixel_dict[sp_val][3] += img_rgb[i][j][0]
            super_pixel_dict[sp_val][4] += img_rgb[i][j][1]
            super_pixel_dict[sp_val][5] += img_rgb[i][j][2]

# print(super_pixel_dict)

# Now we have to calculate mean location and color for each superpixel.
for k in super_pixel_dict.keys():
    for i in range(1,6):
        super_pixel_dict[k][i] = super_pixel_dict[k][i]/super_pixel_dict[k][0]
    
# print(super_pixel_dict)

denom = (math.sqrt(m**2 + n**2))

sal_vals = {}
for sp_key in tqdm(super_pixel_dict.keys()):

    sal_val = 0 

    for sp_key_2 in super_pixel_dict.keys():
        # Euclidean distance between colours of the two superpixels
        r1, g1, b1 = super_pixel_dict[sp_key][3], super_pixel_dict[sp_key][4], super_pixel_dict[sp_key][5]
        r2, g2, b2 = super_pixel_dict[sp_key_2][3], super_pixel_dict[sp_key_2][4], super_pixel_dict[sp_key_2][5]
        color_l2_norm = math.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
        # Euclidean distance between location of the two superpixels
        x1, y1 = super_pixel_dict[sp_key][1], super_pixel_dict[sp_key][2]
        x2, y2 = super_pixel_dict[sp_key_2][1], super_pixel_dict[sp_key_2][2]
        loc_l2_norm = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        sal_val += color_l2_norm * math.exp(-1*loc_l2_norm/denom)
        
    sal_vals[sp_key] = sal_val

# for i in range(len(sal_vals)):
    # print(i, sal_vals[i])

# Normalizing the saliency values to fit range
min_sal_val = 1e10
max_sal_val = 0
for i in sal_vals.keys():
    if sal_vals[i] < min_sal_val:
        min_sal_val = sal_vals[i]
    if sal_vals[i] > max_sal_val:
        max_sal_val = sal_vals[i]
# print(min_sal_val, max_sal_val)

for i in sal_vals.keys():
    sal_vals[i] = (sal_vals[i] - min_sal_val)/(max_sal_val - min_sal_val)
    sal_vals[i] = int(255*sal_vals[i])

# Generating the final saliency map
saliency_img = np.zeros((m,n), np.uint8)
for i in range(m):
    for j in range(n):
        saliency_img[i][j] = sal_vals[super_pixels_labels[i][j]]

# cv2.imshow('Saliency Map', saliency_img)
cv2.imwrite('./output/Q1/'+str(region_size)+'_'+str(int(ruler))+'.jpg', saliency_img)
# cv2.waitKey(0)
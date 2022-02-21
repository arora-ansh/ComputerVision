import numpy as np
import cv2
import math 
from copy import *
from tqdm import tqdm 
from matplotlib import pyplot as plt
import os
from scipy.stats import norm
from skimage import measure
import pandas as pd

def load_images_from_folder(folder):
    images = []
    print("Loading images from folder: ",folder)
    for filename in tqdm(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename),0) #Inputting images in grayscale
        if img is not None:
            images.append(img)
    return images

dl_sal_images = load_images_from_folder("./data/duts_dl_saliency")
non_dl_sal_images = load_images_from_folder("./data/duts_non_dl_saliency")

def separation_measure(img,id=0,gamma=1):
    retval, thresh_sal_img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    #thresh_sal_img holds the otsu thresholded saliency map 
    zero_pixeled_vals = []
    non_zero_pixeled_vals = []
    for i in range(len(thresh_sal_img)):
        for j in range(len(thresh_sal_img[0])):
            if thresh_sal_img[i][j] == 0:
                zero_pixeled_vals.append(img[i][j])
            else:
                non_zero_pixeled_vals.append(img[i][j])
    zero_pixeled_vals = np.array(zero_pixeled_vals)
    non_zero_pixeled_vals = np.array(non_zero_pixeled_vals)
    zero_pixeled_vals = zero_pixeled_vals/255
    non_zero_pixeled_vals = non_zero_pixeled_vals/255
    # We will now find the mean for zero pixeled values and non zero pixeled values
    mean_background = np.mean(zero_pixeled_vals)
    mean_foreground = np.mean(non_zero_pixeled_vals)
    # We will now find the standard deviation for zero pixeled values and non zero pixeled values
    std_background = np.std(zero_pixeled_vals)
    std_foreground = np.std(non_zero_pixeled_vals)

    #We will now calculate the values of z* using the given formula
    z_star = (mean_background*(std_foreground**2)-mean_foreground*(std_background**2))/(std_foreground**2-std_background**2)
    val1 = ((std_foreground*std_background)/(std_foreground**2-std_background**2))
    val2 = math.sqrt((mean_foreground-mean_background)**2 - 2*(std_foreground**2 - std_background**2)*(math.log(std_background)-math.log(std_foreground)))
    z_star1 = z_star + val1*val2
    z_star2 = z_star - val1*val2
    z_star = max(z_star1,z_star2)

    ls = (norm.cdf((z_star-mean_foreground)/std_foreground) - norm.cdf((0-mean_foreground)/std_foreground)) + (norm.cdf((1-mean_background)/std_background) - norm.cdf((z_star-mean_background)/std_background))
    return 1/(1+math.log10(1+gamma*ls))

def concentration_measure(img,id=0):
    retval, thresh_sal_img = cv2.threshold(img, 122, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    #thresh_sal_img holds the otsu thresholded saliency map 
    labeled_img = measure.label(thresh_sal_img,connectivity=2)
    #labeled_img holds the labeled image with connected components, we will now calculate the area of each connected component having value 1 in the thresholded image
    area_dict = {}
    total_one_area = 0
    for i in range(len(labeled_img)):
        for j in range(len(labeled_img[0])):
            if thresh_sal_img[i][j] > 0:
                total_one_area += 1
                if labeled_img[i][j] not in area_dict.keys():
                    area_dict[labeled_img[i][j]] = 1
                else:
                    area_dict[labeled_img[i][j]] += 1
    
    # We will now find the largest area from the dictionary, and assign it to the c_star variable
    # print(area_dict)
    c_star = max(area_dict.values())
    c_star /= total_one_area

    return c_star + (1-c_star)/len(area_dict.keys())

csv_save_data = pd.DataFrame(columns=['Image_ID','DL_Separation_Measure','DL_Concentration_Measure','Non_DL_Separation_Measure','Non_DL_Concentration_Measure'])

print(len(dl_sal_images),len(non_dl_sal_images))

sm1_avg = 0
sm2_avg = 0
cm1_avg = 0
cm2_avg = 0

for i in tqdm(range(len(dl_sal_images))):
    sm1 = separation_measure(dl_sal_images[i])
    sm2 = separation_measure(non_dl_sal_images[i])
    cm1 = concentration_measure(dl_sal_images[i])
    cm2 = concentration_measure(non_dl_sal_images[i])
    new_row = {'Image_ID':i,'DL_Separation_Measure':sm1,'DL_Concentration_Measure':cm1,'Non_DL_Separation_Measure':sm2,'Non_DL_Concentration_Measure':cm2}
    csv_save_data = csv_save_data.append(new_row,ignore_index=True)
    sm1_avg += sm1
    sm2_avg += sm2
    cm1_avg += cm1
    cm2_avg += cm2

sm1 = sm1_avg/len(dl_sal_images)
sm2 = sm2_avg/len(non_dl_sal_images)
cm1 = cm1_avg/len(dl_sal_images)
cm2 = cm2_avg/len(non_dl_sal_images)
print("Average Separation Measure for DL Saliency Maps: ",sm1)
print("Average Separation Measure for Non DL Saliency Maps: ",sm2)
print("Average Concentration Measure for DL Saliency Maps: ",cm1)
print("Average Concentration Measure for Non DL Saliency Maps: ",cm2)
csv_save_data.to_csv("./output/duts_saliency_measures.csv")

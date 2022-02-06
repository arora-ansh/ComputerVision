from turtle import color
import numpy as np
import cv2
import math 
from copy import *
from sympy import resultant
from tqdm import tqdm 
from matplotlib import pyplot as plt

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
    cv2.imwrite("kmeans_tree.png", resultant_img)
    return resultant_img

def eqn3():

    #We first take in the image in color format, with the image being reduced to 85 using kmeans function defined above, Run this if kmeans_leaf.png doesnt exist
    # img = cv2.imread("./leaf.png")
    # cv2.imshow("Original Image",img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = kmeans_segmenter(img, 85)

    #If above code has been run and kmeans_leaf.png exists, then run this
    img = cv2.imread("./data/kmeans_leaf.png")
    # cv2.imshow("85 color Image",img)

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
    # plt.bar(range(len(img_pixel_vals)), color_freq)
    # plt.show()

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

    saliency_min = np.amin(saliency_img)
    saliency_max = np.amax(saliency_img)
    # cv2.imshow("Saliency Image Eqn3", saliency_img)
    cv2.imwrite("./output/saliency_leaf_eqn3.png", 255*((saliency_img-saliency_min)/(saliency_max-saliency_min)))

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def eqn5():
    #We first take in the image in color format, with the image being reduced to 85 using kmeans function defined above, Run this if kmeans_tree.png doesnt exist
    # img = cv2.imread("./BigTree.jpeg")
    # cv2.imshow("Original Image",img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = kmeans_segmenter(img, 85)

    #If above code has been run and kmeans_leaf.png exists, then run this
    color_img = cv2.imread("./data/kmeans_tree.png")
    # cv2.imshow("85 color Image",color_img)

    reg_img = cv2.imread("./data/bigtree_segged.png") #Image segmented by PEGBIS code
    # cv2.imshow("Segmented Image",reg_img)

    #We now need to calculate the color distance metric for each pixel in the image. For this we simply find the distance between two color values used in the image, c_l and c_j.
    m = len(reg_img)
    n = len(reg_img[0])

    #We first lay out all the pixel values in a single set
    region_color_vals = set()
    for i in range(m):
        for j in range(n):
            region_color_vals.add(tuple(reg_img[i][j]))

    print(len(region_color_vals)) #This should be 5 for image the region-ed image

    #Creating an indexed dictionary and list of all the colors in the region image
    region_color_dict = {}
    region_color_list = []
    for i in region_color_vals:
        region_color_list.append(i)
        region_color_list[-1] = np.array(region_color_list[-1])
        region_color_dict[i] = len(region_color_list) - 1

    #Now similarly, we calculate the sets for the 85 color values stored in the image
    color_vals = set()
    for i in range(m):
        for j in range(n):
            color_vals.add(tuple(color_img[i][j]))

    print(len(color_vals)) #This should be 85 for the inputter color image

    #Creating an indexed dictionary and list of all the colors in the color image
    color_dict = {}
    color_list = []
    for i in color_vals:
        color_list.append(i)
        color_list[-1] = np.array(color_list[-1])/255
        color_dict[i] = len(color_list) - 1

    #Now, we will create a color euclidean distance dictionary for each pair of the 85 colors in the color image
    color_dist = {}
    for i in range(len(color_list)):
        for j in range(len(color_list)):
            color_dist[(i,j)] = math.sqrt(sum((color_list[i] - color_list[j])**2))

    # print(color_dist)

    #Now, in this case we need to take out the distance between individual regions by the formula - sum of sum of f(c1i)f(c2j)D(c1i,c2j)
    #Here f(c1i) signifies probability of color ci in region 1, and D denotes the color distance metric calculated above
    #For this we first need to calculate the probability of each color in each region given by the region image

    regions_freq = {}
    #Will initiate the dictionary value as 0 for all colours for all regions 
    for i in range(len(region_color_list)):
        regions_freq[i] = {}

    for i in range(len(region_color_list)):
        for j in range(len(color_list)):
            regions_freq[i][j] = 0

    region_count = np.zeros((len(region_color_list)))
    for i in range(len(reg_img)):
        for j in range(len(reg_img[0])):
            #We will find out the region from the colour stored at reg_img[i][j]
            region_no = region_color_dict[tuple(reg_img[i][j])]
            region_count[region_no] += 1
            color_no = color_dict[tuple(color_img[i][j])]
            regions_freq[region_no][color_no] += 1

    # Finding probability of each color in that particular region by dividing the regions_freq for that region by the region_count

    for i in range(len(region_count)):
        for j in range(len(color_list)):
            regions_freq[i][j] = regions_freq[i][j]/region_count[i]

    # print(regions_freq)
    # print(region_count)

    # We will now calculate the region distance between all combinations of regions, and store it as a dictionary of pairs
    region_dist = {}
    for i in tqdm(range(len(region_color_list))):
        for j in range(len(region_color_list)):
            if i==j:
                region_dist[(i,i)] = 0
            else:
                region_dist[(i,j)] = 0
                for k in range(len(color_list)):
                    for l in range(len(color_list)):
                        region_dist[(i,j)] += color_dist[(k,l)]*regions_freq[i][k]*regions_freq[j][l]

    region_weight = []
    total_pixels = sum(region_count)
    for i in region_count:
        region_weight.append(i/total_pixels)

    saliency_vals = np.zeros((len(region_color_list)))
    for i in range(len(region_color_list)):
        for j in range(len(region_color_list)):
            if i != j:
                saliency_vals[i] += region_weight[j]*region_dist[(i,j)]

    saliency_img = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            saliency_img[i][j] = saliency_vals[region_color_dict[tuple(reg_img[i][j])]]

    saliency_min = np.amin(saliency_img)
    saliency_max = np.amax(saliency_img)
    # cv2.imshow("Saliency Image",saliency_img)
    cv2.imwrite("./output/saliency_tree_eqn5.png",255*((saliency_img-saliency_min)/(saliency_max-saliency_min)))

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    print("1. Equation 3\n2. Equation 5")
    choice = int(input("Enter your choice: "))
    if choice == 1:
        eqn3()
    elif choice == 2:
        eqn5()
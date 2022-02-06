from turtle import width
import numpy as np
import cv2
import math 
from copy import *
from tqdm import tqdm 
from matplotlib import pyplot as plt

#We first capture the video and convert it into frames and store the frames in a folder.
vidcap = cv2.VideoCapture("./data/shahar_walk.avi")
frame_count = 0 #Number of frames in the video stored in this variable.
frames = []

vid_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(vid_width, vid_height)

while vidcap.isOpened():
    flag, image = vidcap.read()
    if not flag:
        break
    frame_count += 1
    cv2.imwrite("./data/frames/%d.png" % frame_count, image)
    frames.append(image)

vidcap.release()
print("Number of frames in the video: ", frame_count)
#All captured images are now stored in the frames folder. 

#Now we will find the pixel level median of all the frames.
cv2.imwrite("./data/median.png",np.median(frames, axis=0))
median = cv2.imread("./data/median.png")
# cv2.imshow('Median Image', median)

#We will now perform absolute background subtraction for each frame and store the result in a folder.
#We then will find the ROI of each frame and draw a bounding circle around it. We will then stitch the images together into a video.
for f in tqdm(range(frame_count)):
    abs_sub = cv2.absdiff(frames[f], median)
    cv2.imwrite("./data/abs_sub/%d.png" % (f+1), abs_sub)
    abs_sub_gs = cv2.cvtColor(abs_sub, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(abs_sub_gs, 10, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite("./data/thresh/%d.png" % (f+1), thresh)
    #Find the ROI of the given thresholded image.
    thresh_img = cv2.imread("./data/thresh/%d.png" % (f+1),0)
    width_left = vid_height
    width_right = -1
    height_bottom = vid_height
    height_top = -1
    for i in range(vid_width):
        for j in range(vid_height):
            if thresh_img[j][i] == 255:
                if i < width_left:
                    width_left = i
                if i > width_right:
                    width_right = i
                if j < height_bottom:
                    height_bottom = j
                if j > height_top:
                    height_top = j
    if width_right-width_left > height_top-height_bottom:
        radius = int((width_right-width_left)/2)
        center = int((width_left+width_right)/2), int((height_top+height_bottom)/2)
    else:
        radius = int((height_top-height_bottom)/2)
        center = int((width_left+width_right)/2), int((height_top+height_bottom)/2)
    
    #Drawing bounding circle on the ROI
    circled_image = cv2.circle(frames[f],center,radius,(0,0,255),2)
    cv2.imwrite("./data/circled/%d.png" % (f+1), circled_image)
    
#We will now stitch the images stored in circled together into a video.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('./output/median_video.avi', fourcc, 24, (vid_width,vid_height))
for i in tqdm(range(frame_count)):
    video.write(cv2.imread("./data/circled/%d.png" % (i+1)))

video.release()

# cv2.waitKey(0)
# cv2.destroyAllWindows()
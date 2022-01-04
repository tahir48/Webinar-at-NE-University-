# Real-Time-Raised-Hand-Detection-Using-SIFT-Features

## Table Of Contents
•	General Info
•	Technologies
•	Setup
### General Info
This funny project is a primitive raised hand detection using SIFT features, need some improvement, indeed.
This python code uses OPENCV module to extract and match sift features.
The function sift_matcher requires two input images, one for template, another to check how many descriptors matches with the template and normal image. It returns keypoints on each image for those matches, so that they can be used to draw on real-time camera window.
Once number of matches exceeds a predefined threshold, “Hand Raised” warning appears on the screen.
### Technologies
1.	opencv-python==4.5.4.60 
2.	opencv-contrib-python==4.5.4.60 
3.	numpy==1.21.4
### Setup
Make sure opencv, opencv-contrib-python modules are installed. If not,  -pip install opencv-python and  -pip install opencv-contrib-python helps.
Enjoy


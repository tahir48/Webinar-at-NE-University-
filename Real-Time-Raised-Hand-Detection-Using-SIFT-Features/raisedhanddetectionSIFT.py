import cv2
import numpy as np
import keyboard

def sift_detector(new_image, image_template):
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    image2 = image_template
    
    sift = cv2.xfeatures2d.SIFT_create()

    # Obtain the keypoints and descriptors using SIFT
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)

    return keypoints_1,keypoints_2,matches

cap = cv2.VideoCapture(0)
image_template = cv2.imread('C:/Users/Lenovo/Desktop/handd.png', 0) 

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    top_left_x = int(width / 3)
    top_left_y = int((height / 2) + (height / 4))
    bottom_right_x = int((width / 3) * 2)
    bottom_right_y = int((height / 2) - (height / 4))
    


    cv2.rectangle(frame, (top_left_x,top_left_y), (int(bottom_right_x),int(bottom_right_y)), 255, 3)
    
    cropped = frame[bottom_right_y:top_left_y , top_left_x:bottom_right_x]
    
    frame = cv2.flip(frame,1)
    
    # Get number of SIFT matches
    keypoints_1,keypoints_2,matches = sift_detector(cropped, image_template)
    img3 = cv2.drawMatches(frame, keypoints_1, image_template, keypoints_2, matches[:25], cropped, flags=2)
    print("matches",len(matches))

    if len(matches) > 40:
        #cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), (0,255,0), 3)
        cv2.putText(img3,'Hand Found',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2)

    cv2.imshow("frame",img3)
    cv2.waitKey(1)
    if len(matches) > 40:
        cv2.imwrite("result.jpg",img3)
    if keyboard.is_pressed("q"):
        print("q is pressed, loop is ending")
        break

cap.release()
cv2.destroyAllWindows()   


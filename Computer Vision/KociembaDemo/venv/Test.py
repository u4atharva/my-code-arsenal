import cv2
import numpy as np
 
def empty(a):
    pass

cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
 
while True:
    success, img = cap.read()

    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
 
    cv2.imshow("Gray",imgGray)
    cv2.waitKey(1)
import cv2 as cv
import numpy as np
import time
import requests
import imutils


def empty(a):
	pass

cv.namedWindow("TrackBar")
cv.resizeWindow("TrackBar", 640, 240)
cv.createTrackbar("H Min", "TrackBar", 0, 255, empty)
cv.createTrackbar("S Min", "TrackBar", 0, 255, empty)
cv.createTrackbar("V Min", "TrackBar", 0, 255, empty)

cv.createTrackbar("H Max", "TrackBar", 0, 255, empty)
cv.createTrackbar("S Max", "TrackBar", 0, 255, empty)
cv.createTrackbar("V Max", "TrackBar", 0, 255, empty)

if __name__ == "__main__":

    url = "http://192.168.1.16:8080/shot.jpg"
    frame_width = 640
    frame_height = 480
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv.imdecode(img_arr, -1)
        img = imutils.resize(img, width=frame_width, height=frame_height)

        # face = cv.imread('Images/Cube_Face.jpg')

        half = cv.resize(img, (0, 0), fx = 0.5, fy = 0.5)
        img_HSV = cv.cvtColor(half, cv.COLOR_BGR2HSV)
        h_min = cv.getTrackbarPos("H Min", "TrackBar")
        h_max = cv.getTrackbarPos("H Max", "TrackBar")
        s_min = cv.getTrackbarPos("S Min", "TrackBar")
        s_max = cv.getTrackbarPos("S Max", "TrackBar")
        v_min = cv.getTrackbarPos("V Min", "TrackBar")
        v_max = cv.getTrackbarPos("V Max", "TrackBar")

        # print(h_min,h_max,s_min,s_max,v_min,v_max)
        print(h_min,s_min,v_min,h_max,s_max,v_max)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv.inRange(img_HSV, lower, upper)
        #img_result = cv.bitwise_and(half, half, mask=mask)

        cv.imshow("Original", half)
        # cv.imshow("HSV", img_HSV)
        cv.imshow("Mask", mask)
        # cv.imshow("Result", img_result)


        cv.waitKey(1)
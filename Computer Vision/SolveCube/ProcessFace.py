_author_ = 'compiler'

import cv2 as cv


def empty(a):
    pass

cv.namedWindow("TrackBar")
cv.resizeWindow("TrackBar", 640,240)
cv.createTrackbar("H Min", "TrackBar", 0, 255, empty)
cv.createTrackbar("H Max", "TrackBar", 0, 255, empty)

while True:
    img = cv.imread("Images/1.jpg")
    half_img = cv.resize(img, (0,0), fx = 0.5, fy = 0.5)
    img_HSV = cv.cvtColor(half_img, cv.COLOR_BGR2HSV)
    cv.imshow("HSV", img_HSV)

    h_min = cv.getTrackbarPos("TrackBar", "H Min")
    h_max = cv.getTrackbarPos("TrackBar", "H Max")
    print(str(h_min) + "     " + str(h_max))

    cv.waitKey(1)

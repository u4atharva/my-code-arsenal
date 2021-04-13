import cv2 as cv
import numpy as np
import time

while True:

    # define the list of boundaries (Readings received from CalibrateColors.py app)
    boundaries = [
        ([138, 119, 201], [179, 184, 230]), # red: 1
        ([100, 196, 129], [107, 248, 255]), # blue: 2
        ([5, 118, 221], [13, 144, 255]), # orange: 3
        ([104, 38, 217], [110, 55, 240]), # white: 4
        ([69, 175, 198], [82, 234, 242]), # green: 5
        ([23, 58, 196], [41, 100, 220]) # yellow: 6
    ]
    counter = 0

    face = cv.imread('Images/Cube_Face.jpg')
    half = cv.resize(face, (0, 0), fx = 0.5, fy = 0.5)
    img_HSV = cv.cvtColor(half, cv.COLOR_BGR2HSV)

    # loop over the boundaries
    for (lower, upper) in boundaries:
        counter += 1
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        # find the colors within the specified boundaries and apply the mask
        mask = cv.inRange(img_HSV, lower, upper)
        output = cv.bitwise_and(half, half, mask = mask)
        # show the images
        cv.imshow("images_{counter}".format(counter = counter), np.hstack([half, output]))

    cv.waitKey(0)
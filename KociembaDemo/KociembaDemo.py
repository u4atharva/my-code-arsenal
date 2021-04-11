import kociemba
import cv2 as cv
import numpy as np
import time

#solve_sequence = kociemba.solve('DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD')

#print(solve_sequence)

def empty(a):
	pass

cv.namedWindow("TrackBar")
cv.resizeWindow("TrackBar", 640, 240)
cv.createTrackbar("H Min", "TrackBar", 0, 179, empty)
cv.createTrackbar("H Max", "TrackBar", 19, 179, empty)
cv.createTrackbar("S Min", "TrackBar", 110, 255, empty)
cv.createTrackbar("S Max", "TrackBar", 240, 255, empty)
cv.createTrackbar("V Max", "TrackBar", 153, 255, empty)
cv.createTrackbar("V Min", "TrackBar", 255, 255, empty)



while True:
	face = cv.imread('Images/1.jpg')
	half = cv.resize(face, (0, 0), fx = 0.5, fy = 0.5) 
	img_HSV = cv.cvtColor(half, cv.COLOR_BGR2HSV)
	h_min = cv.getTrackbarPos("H Min", "TrackBar")
	h_max = cv.getTrackbarPos("H Max", "TrackBar")
	s_min = cv.getTrackbarPos("S Min", "TrackBar")
	s_max = cv.getTrackbarPos("S Max", "TrackBar")
	v_min = cv.getTrackbarPos("V Min", "TrackBar")
	v_max = cv.getTrackbarPos("V Max", "TrackBar")
	print(h_min,h_max,s_min,s_max,v_min,v_max)
	
	lower = np.array([h_min, s_min, v_min])
	upper = np.array([h_max, s_max, v_max])
	mask = cv.inRange(img_HSV, lower, upper)
	#img_result = cv.bitwise_and(half, half, mask=mask)

	cv.imshow("Original", half)
	cv.imshow("HSV", img_HSV)
	cv.imshow("Mask", mask)
	#cv.imshow("Result", img_result)


	cv.waitKey(1)


#while True:
#	r_min = cv.getTrackbarPos("TrackBar", "R Min")
#	r_max = cv.getTrackbarPos("TrackBar", "R Max")
#	g_min = cv.getTrackbarPos("TrackBar", "G Min")
#	g_max = cv.getTrackbarPos("TrackBar", "G Max")
#	b_min = cv.getTrackbarPos("TrackBar", "B Min")
#	b_max = cv.getTrackbarPos("TrackBar", "B Max")
#	print(r_min)
#	print(r_max)
#	print(g_min)
#	print(g_max)
#	time.sleep(2)

## define the list of boundaries
#boundaries = [
#	([17, 15, 100], [50, 56, 200]),
#	([86, 31, 4], [220, 88, 50]),
#	([b_min, g_min, r_min], [b_max, g_max, r_max]),
#	([150, 150, 150], [255, 255, 255]),
#	([15, 50, 175], [50, 90, 225]),
#	([150, 150, 150], [255, 255, 255])
#]
## red: 1
## blue: 2
## orange: 3
## white: 4
## green: 5
## yellow: 6
#counter = 0

## loop over the boundaries
#for (lower, upper) in boundaries:
#	counter += 1
#	# create NumPy arrays from the boundaries
#	lower = np.array(lower, dtype = "uint8")
#	upper = np.array(upper, dtype = "uint8")
#	# find the colors within the specified boundaries and apply the mask
#	mask = cv.inRange(half, lower, upper)
#	output = cv.bitwise_and(half, half, mask = mask)
#	# show the images
#	if counter == 3:
#		cv.imshow("images_{counter}".format(counter = counter), np.hstack([half, output]))

#cv.waitKey(0)

import cv2
import numpy as np

frameWidth = 640
frameHeight = 480

# cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)

def empty(a):
    pass

def getContours(img, imgContour):

    contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x=0
    y=0
    w=0
    h=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 90000:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(len(approx))
            # if len(approx) == 4:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            x, y, w, h = cv2.boundingRect(approx)
            # (x, y) , (x+w, y+h)
            # (x,y) , (x+w, y), (x+w, y+h), (x, y+h)
            print(x,y,w,h)

    return x, y, w, h


img = cv2.imread('Images/Cube_Face.jpg')
imgContour = img.copy()

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 71, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 100, 255, empty)

while True:
    # success, img = cap.read()

    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    t1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    t2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    x, y, w, h = getContours(imgDil, imgContour)

    top_left_x = min([x, x + w])
    top_left_y = min([y, y + h])
    bot_right_x = max([x, x + w])
    bot_right_y = max([y, y + h])
    imgCrop = img[top_left_y:bot_right_y, top_left_x:bot_right_x]

    cv2.imshow("Result", img)
    # cv2.imshow("Blur", imgBlur)
    # cv2.imshow("Gray", imgGray)
    # cv2.imshow("Canny", imgCanny)
    # cv2.imshow("Dialation", imgDil)
    cv2.imshow("Contour", imgContour)
    cv2.imshow("Cropped", imgCrop)

    if cv2.waitKey(1) & 0xff== ord('q'):
        break


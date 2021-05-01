import cv2
import numpy as np
import ClassifyColors
from ClassifyColors import get_color, get_cell
import requests
import imutils
import FindCube

url = "http://192.168.1.16:8080/shot.jpg"

def empty(a):
    pass


def detect_cube(imgDil, imgContour, img):

    contours, heirarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x=0
    y=0
    w=0
    h=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # if len(approx) == 4:
            cv2.drawContours(imgContour, cnt, -1, (0, 255, 0), 3)
            x, y, w, h = cv2.boundingRect(approx)
    top_left_x = min([x, x + w])
    top_left_y = min([y, y + h])
    bot_right_x = max([x, x + w])
    bot_right_y = max([y, y + h])
    return img[top_left_y:bot_right_y, top_left_x:bot_right_x]

frameWidth = 640
frameHeight = 480

# cap = cv2.VideoCapture(0)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# img = cv2.imread('Images/Cube_Face.jpg')


cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 71, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 41, 255, empty)

if __name__ == "__main__":
    while True:

        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=frameWidth, height=frameHeight)

        # success, img = cap.read()

        imgContour = img.copy()
        imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

        t1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        t2 = cv2.getTrackbarPos("Threshold2", "Parameters")

        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

        imgCrop = FindCube.detect_cube(imgDil, imgContour, img)
        # matrix = ClassifyColors.create_color_matrix(imgCrop)

        if imgCrop.size == 0:
            pass
        else:
            cv2.imshow("Result", imgCrop)
            matrix = ClassifyColors.create_color_matrix(imgCrop)
        # cv2.imshow("Blur", imgBlur)
        # cv2.imshow("Gray", imgGray)
        # cv2.imshow("Canny", imgCanny)
        # cv2.imshow("Dialation", imgDil)
        cv2.imshow("Contour", imgContour)
        # cv2.imshow("Cropped", imgCrop)
        # cv2.imshow("1:1", img_temp)

        if cv2.waitKey(1) & 0xff== ord('q'):
            break


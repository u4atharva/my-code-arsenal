import cv2
import numpy as np
import ClassifyColors
import requests
import imutils
from ClassifyColors import get_color, get_cell

url = "http://192.168.1.16:8080/shot.jpg"

def empty(a):
    pass


def detect_cube(imgDil, imgContour, img):

    contours, heirarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours, heirarchy = cv2.findContours(imgDil, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    x,y,w,h=0,0,0,0

    for c in contours:
        # hull = cv2.convexHull(c)
        area = cv2.contourArea(c)
        if area > 10000 and area < 100000:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if len(approx) == 4:
            cv2.drawContours(imgContour, c, -1, (0, 255, 0), 2)
            # cv2.drawContours(imgContour, [hull], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(approx)

    # for i in range(numcards):
    #     card = contours[i]
    #     peri = cv2.arcLength(card, True)
    #     approx = cv2.approxPolyDP(card, 0.02 * peri, True)
    #     rect = cv2.minAreaRect(contours[i])
    #     r = cv2.cv.BoxPoints(rect)
    #
    #     h = np.array([[0, 0], [399, 0], [399, 399], [0, 399]], np.float32)
    #     approx = np.array([item for sublist in approx for item in sublist], np.float32)
    #     transform = cv2.getPerspectiveTransform(approx, h)
    #     warp[i] = cv2.warpPerspective(img, transform, (400, 400))


    top_left_x = min([x, x + w])
    top_left_y = min([y, y + h])
    bot_right_x = max([x, x + w])
    bot_right_y = max([y, y + h])
    return img[top_left_y:bot_right_y, top_left_x:bot_right_x]


cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 71, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 41, 255, empty)

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
        imgContour2 = img.copy()
        imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        imAdapTh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        t1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        t2 = cv2.getTrackbarPos("Threshold2", "Parameters")

        print(t1, t2)

        imgCanny = cv2.Canny(imgGray, t1, t1)
        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

        detect_cube(imgDil, imgContour, img)
        # print(imgCrop)
        # matrix = ClassifyColors.create_color_matrix(imgCrop)

        imgCanny = cv2.Canny(imAdapTh, t1, t1)
        kernel = np.ones((5, 5))
        imgDil2 = cv2.dilate(imgCanny, kernel, iterations=1)
        detect_cube(imgDil2, imgContour2, imAdapTh)
        res = np.hstack((imgContour, imgContour2))

        imgBlur = cv2.GaussianBlur(imAdapTh, (7, 7), 1)
        # cv2.imshow("AT", imgBlur)
        cv2.imshow("Result", imgContour)

        if cv2.waitKey(1) & 0xff== ord('q'):
            break


import cv2
import numpy as np
import ClassifyColors
import requests
import imutils
from ClassifyColors import get_color, get_cell
from scipy.interpolate import splprep, splev

url = "http://192.168.1.16:8080/shot.jpg"

def empty(a):
    pass


def detect_cube(imgDil, imgContour, img):

    contours, heirarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours, heirarchy = cv2.findContours(imgDil, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    x1,y1,w,h=0,0,0,0

    for c in contours:
        # hull = cv2.convexHull(c)
        area = cv2.contourArea(c)
        if area > 10000 and area < 100000:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if len(approx) == 4:
            cv2.drawContours(imgContour, c, -1, (0, 255, 0), 2)
            # cv2.drawContours(imgContour, [hull], -1, (0, 255, 0), 2)
            x1, y1, w, h = cv2.boundingRect(approx)

    smoothened = []
    for contour in contours:
        x, y = contour.T

        print((x))
        print((y))
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        # tck, u = splprep([x, y], u=None, s=1.0, per=1)
        tck, u = splprep([x, y], u=None, s=0, per=1)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        u_new = np.linspace(u.min(), u.max(), 25)
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
        area = cv2.contourArea(contour)
        if area > 10000 and area < 100000:
            smoothened.append(np.asarray(res_array, dtype=np.int32))

    # Overlay the smoothed contours on the original image
    cv2.drawContours(imgContour, smoothened, -1, (255, 255, 255), 2)


    top_left_x = min([x1, x1 + w])
    top_left_y = min([y1, y1 + h])
    bot_right_x = max([x1, x1 + w])
    bot_right_y = max([y1, y1 + h])
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


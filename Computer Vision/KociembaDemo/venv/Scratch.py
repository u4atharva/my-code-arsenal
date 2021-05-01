import cv2
import numpy as np
import requests
import imutils
import ClassifyColors


def get_cube(img, cThr=[50,100]):
    top_left_x, top_left_y, bot_right_x, bot_right_y = 0,0,0,0

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur,cThr[0],cThr[1])
    # kernel = np.ones((5,5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgErod = cv2.erode(imgDial, kernel, iterations=2)
    closing = cv2.morphologyEx(imgErod, cv2.MORPH_CLOSE, kernel)

    adThresh = cv2.adaptiveThreshold(closing, 20, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 0)

    try:
        _, contours, hierarchy = cv2.findContours(adThresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    except:
        contours, hierarchy = cv2.findContours(adThresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        area = cv2.contourArea(c)
        cv2.drawContours(img, c, -1, (255, 255, 255), 2)
        if area > 30000 and area < 100000:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                cv2.drawContours(img, c, -1, (0, 255, 0), 7)
                x, y, w, h = cv2.boundingRect(approx)
                top_left_x = min([x, x + w])
                top_left_y = min([y, y + h])
                bot_right_x = max([x, x + w])
                bot_right_y = max([y, y + h])
        cv2.imshow("Result", img)
    success = False

    if top_left_y and bot_right_y and top_left_x and bot_right_x:
        success = True
    return success, img[top_left_y:bot_right_y, top_left_x:bot_right_x]

def main():

    url = "http://192.168.1.16:8080/shot.jpg"
    frame_width = 640
    frame_height = 480
    while True:

        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=frame_width, height=frame_height)

        success, imgCube = get_cube(img)

        if success:
            matrix = ClassifyColors.create_color_matrix(imgCube)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

if __name__ == "__main__": main()


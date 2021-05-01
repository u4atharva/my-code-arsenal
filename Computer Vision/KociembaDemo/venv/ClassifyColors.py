import cv2 as cv
import numpy as np
import time

# define the list of boundaries (Readings received from CalibrateColors.py app)
# boundaries = [
#     ([138, 119, 201], [179, 184, 230]), # red: 1
#     ([100, 196, 129], [107, 248, 255]), # blue: 2
#     ([5, 118, 221], [13, 144, 255]), # orange: 3
#     ([104, 38, 217], [110, 55, 240]), # white: 4
#     ([69, 175, 198], [82, 234, 242]), # green: 5
#     ([23, 58, 196], [41, 100, 220]) # yellow: 6
# ]

boundaries = [
    ([137, 117, 200], [178, 183, 233]), # red: 1
    ([100, 196, 129], [107, 248, 255]), # blue: 2
    ([0, 72, 20], [20, 190, 255]), # orange: 3
    ([86, 0, 0], [148, 255, 255]), # white: 4
    ([55, 122, 0], [140, 202, 253]), # green: 5
    ([20, 0, 143], [60, 146, 224]) # yellow: 6
]


def get_color(imgInp):
    imgHSV = cv.cvtColor(imgInp, cv.COLOR_BGR2HSV)

    hsv_values = np.mean(imgHSV, axis=(0, 1))

    high = hsv_values[0]
    sat = hsv_values[1]
    val = hsv_values[2]

    # loop over the boundaries
    counter = 1
    for (lower, upper) in boundaries:

        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        if (high > lower[0] and high < upper[0]) and (sat > lower[1] and sat < upper[1]) and (val > lower[2] and val < upper[2]):
            break
        counter += 1

    if counter == 7:     # unable to match any range
        pass            # To-Do add error margin code

    return counter


def get_color_detail(list):
    for n, i in enumerate(list):
        if i == 1:
            list[n] = 'Red'
        elif i == 2:
            list[n] = 'Blue'
        elif i == 3:
            list[n] = 'Orange'
        elif i == 4:
            list[n] = 'White'
        elif i == 5:
            list[n] = 'Green'
        elif i == 6:
            list[n] = 'Yellow'

    return list

def create_color_matrix(img):
    matrix = []
    pass
    for i in range(0,3):
        for j in range(0,3):
            img_temp = get_cell(img, i, j)
            # cv.imwrite(str(i)+'_'+str(j)+'.png', img_temp)
            matrix.append(get_color(img_temp))

    print(get_color_detail(matrix))


def get_cell(img, row, col):
    height, width = img.shape[:2]
    if width > height:
        dim = (height * 5, height * 5)
    else:
        dim = (width * 5, width * 5)

    # resize image - upscaling and converting to square
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    height, width = resized.shape[:2]

    cropped_y_start = int((height * 0.33) * row)
    cropped_y_end = int((height * 0.33) + (height * 0.33 * row))
    cropped_x_start = int((width * 0.33) * col)
    cropped_x_end = int((width * 0.33) + (width * 0.33) * col)

    img_cell = resized[cropped_y_start:cropped_y_end, cropped_x_start:cropped_x_end]

    height, width = img_cell.shape[:2]
    cropped_y_start = int((height * 0.25))
    cropped_y_end = int((height * 0.50))
    cropped_x_start = int((width * 0.25))
    cropped_x_end = int((width * 0.50))

    img_cell = img_cell[cropped_y_start:cropped_y_end, cropped_x_start:cropped_x_end]
    return img_cell


if __name__ == "__main__":

    while True:

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

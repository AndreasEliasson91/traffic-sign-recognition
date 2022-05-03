import cv2
import numpy as np


def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y],
                                                 (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None,
                                                 scale,
                                                 scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows

        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x],
                                          (img_array[0].shape[1], img_array[0].shape[0]),
                                          None,
                                          scale,
                                          scale)
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(img_array)
        ver = hor
    return ver


def main():
    img = cv2.imread('./img/house.jpg')
    img2 = cv2.imread('./img/donald.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    stacked_imgs = stack_images(0.5, ([img, img2], [img_gray, img_gray2]))
    # horizontal = np.hstack((img, img))
    # vertical = np.vstack((img, img))

    cv2.imshow("Stacked 1", stacked_imgs)
    #cv2.imshow("Stacked 2", vertical)

    cv2.waitKey(0)


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv

from application.controllers.find_contours import get_contours
from application.data.data_utils import stack_images


def pre_processing(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (1, 1), 5)
    img_canny = cv.Canny(img_blur, 150, 200)
    kernel = np.ones((5, 5), np.uint8)
    img_di = cv.dilate(img_canny, kernel, iterations=4)
    img_eroded = cv.erode(img_di, kernel, iterations=4)
    img_contour = img.copy()
    return img_eroded, img_contour


def main():
    in_folder = r'../data/datasets/training_data/training_1.0/training'
    img_count = 0
    img_list = []

    for subdir, dirs, files in os.walk(in_folder):
        for file in files:
            img_count += 1
            img_path = os.path.join(subdir, file)
            print(f'Image number: {img_count} {file}')
            img = cv.imread(img_path)
            img_eroded, img_contour = pre_processing(img)
            get_contours(img_eroded, img_contour)
            img_list.append(img_contour)

    img_stack = stack_images(0.5, img_list)
    cv.imshow('images', img_stack)
    cv.waitKey(0)

    print(f"Total number of images processed : {img_count}")


if __name__ == '__main__':
    main()

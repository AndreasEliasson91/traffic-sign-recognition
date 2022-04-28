import cv2 as cv
import numpy as np
from application.data.data_utils import stack_images


def get_contours(img, img_contour):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    print(f'Hierarchy: {hierarchy}')
    for contour in contours:
        area = cv.contourArea(contour)
        cv.drawContours(img_contour, contour, -1, (255, 0, 0), 3)
        arc_length = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, arc_length * 0.1, True)
        obj_corners = len(approx)
        x, y, w, h = cv.boundingRect(approx)
        # cv.rectangle(img_contour, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if obj_corners == 3:
            object_type = 'Triangle'
        elif obj_corners == 4:
            asp_ratio = w / float(h)
            if 0.98 < asp_ratio < 1.03:
                object_type = 'Square'
            else:
                object_type = 'Rectangle'
        else:
            object_type = 'Circle'

        cv.putText(img_contour, object_type, (x + (w // 2) - len(object_type)*4, y-10), cv.FONT_HERSHEY_COMPLEX,
                   0.5, (255, 0, 0), 1)
        print(f'Object type: {object_type}')
        print(f'Object corners: {obj_corners}')
        print(f'Object area: {area}')
        print(f'Object perimeter: {arc_length}\n')


def main():
    # img = cv.imread('../data/datasets/training_data/training_1.0/training'
    #                 '/prohibitory_signs-e1bdaab4-c228-11ec-acd7-18cc1895e0b0.jpg')
    img = cv.imread('../data/datasets/training_data/training_1.0/training'
                    '/warning_signs-e1b7de53-c228-11ec-8b52-18cc1895e0b0.jpg')
    img_blank = np.zeros_like(img)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (1, 1), 5)
    img_canny = cv.Canny(img_blur, 150, 200)
    kernel = np.ones((5, 5), np.uint8)
    img_di = cv.dilate(img_canny, kernel, iterations=4)
    img_eroded = cv.erode(img_di, kernel, iterations=4)
    img_contour = img.copy()

    get_contours(img_eroded, img_contour)

    img_stack = stack_images(2, ([img, img_gray, img_blur, img_blank],
                                 [img_canny, img_di, img_eroded, img_contour]))

    cv.imshow('images', img_stack)

    cv.waitKey(0)


if __name__ == '__main__':
    main()

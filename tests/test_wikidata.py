import os

import cv2
import numpy as np
import matplotlib.image as mpimg

from application.data.data_utils import stack_images, write_annotations
from application.data.models import CustomImageDataset
from os import listdir, rename
from torchvision import transforms

BROKEN_IMAGES = {
    0: 'A_022.jpg',
    1: 'A_069.jpg',
    2: 'B_010.jpg',
    3: 'B_02.jpg',
    4: 'B_07.jpg',
    5: 'B_08.jpg',
    6: 'E_033.jpg',
    7: 'E_034.jpg',
    8: 'E_035.jpg',
    9: 'E_044.jpg',
    10: 'E_050.jpg',
    11: 'E_051.jpg',
    12: 'E_053.jpg',
    13: 'F_010.jpg',
    14: 'F_013.jpg',
    15: 'F_017.jpg',
    16: 'F_02.jpg',
    17: 'F_022.jpg',
    18: 'F_023.jpg',
    19: 'F_025.jpg',
    20: 'F_028.jpg',
    21: 'F_03.jpg',
    22: 'F_040.jpg',
    23: 'F_041.jpg',
    24: 'F_042.jpg',
    25: 'F_043.jpg',
    26: 'F_044.jpg',
    27: 'F_06.jpg',
    28: 'F_08.jpg',
    29: 'F_09.jpg',
    30: 'G_07.jpg',
    31: 'H_02.jpg',
    32: 'S_010.jpg',
    33: 'S_011.jpg',
    34: 'S_014.jpg',
    35: 'S_03.jpg',
    36: 'S_04.jpg',
    37: 'S_06.jpg',
    38: 'S_07.jpg',
    39: 'S_08.jpg',
    40: 'S_09.jpg',
    41: 'T_010.jpg',
    42: 'T_011.jpg',
    43: 'T_012.jpg',
    44: 'T_013.jpg',
    45: 'T_014.jpg',
    46: 'T_015.jpg',
    47: 'T_016.jpg',
    48: 'T_017.jpg',
    49: 'T_018.jpg',
    50: 'T_019.jpg',
    51: 'T_022.jpg',
    52: 'T_026.jpg',
    53: 'T_037.jpg',
    54: 'T_038.jpg',
    55: 'T_048.jpg',
    56: 'T_05.jpg',
    57: 'T_059.jpg',
    58: 'T_062.jpg',
    59: 'T_063.jpg',
    60: 'T_064.jpg',
    61: 'T_07.jpg',
    62: 'T_08.jpg',
    63: 'T_09.jpg',
    64: 'X_03.jpg',
    65: 'Z_016.jpg',
    66: 'Z_018.jpg',
    67: 'Z_04.jpg',
}

wikipedia_dataset = CustomImageDataset(
    annotations_file=r'../application/data/datasets/training_data/wikipedia_signs/annotation/annotations_wikipedia_data.csv',
    img_dir=r'../application/data/datasets/training_data/wikipedia_signs/'
)

broken_signs = CustomImageDataset(
    annotations_file='./broken_images/annotation/broken_images.csv',
    img_dir='./broken_images/'
)


def main():
    # write_annotations('./broken_images/', './broken_images/annotation/broken_images.csv')
    #
    broken_signs.transform = transforms.ToPILImage()
    wikipedia_dataset.transform = transforms.ToPILImage()

    # broken_signs.plot_images(index=4)
    broken_signs.plot_images(rows=10, cols=7)
    wikipedia_dataset.plot_images(rows=10, cols=7)
    #
    for i in range(len(broken_signs)):
        print(broken_signs[i][0].mode)
    #
    # print('*'*30)
    #
    # for i in range(len(wikipedia_dataset)):
    #     print(wikipedia_dataset[i][0].mode)

    # from PIL import Image
    # for image in listdir('./broken_images/'):
    #     print(image)
    #     if os.path.isfile(f'./broken_images/{image}'):
    #         img = Image.open(f'./broken_images/{image}').convert('RGB')
    #         img.save(f'./broken_images/{image}')
    #         print('done')
    # invalid = []
    # for image in listdir('./broken_images/'):
    #     if image.endswith('.jpg'):
    #         try:
    #             img = Image.open(f'./broken_images/{image}')  # open the image file
    #             img.verify()  # verify that it is, in fact an image
    #         except (IOError, SyntaxError) as e:
    #             invalid.append(image)
    #
    # print([v for v in invalid])

    # img = cv2.imread('./broken_images/B_02.jpg')
    # img_gray = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
    # img_conv = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    # stack = stack_images(1., [img, img_gray, img_conv])
    #
    # cv2.imshow('Images', stack)
    # cv2.waitKey(0)


if __name__ == '__main__':
    main()


import cv2
import numpy as np

from os import listdir, rename
from os.path import isfile, join

LABELS = {
    'A': 'Warning Sign',
    'B': 'Priority Sign',
    'C': 'Prohibitory Sign',
    'D': 'Mandatory Sign',
    'E': 'Instruction Sign',
    'F': 'Location Sign\n(Directions)',
    'G': 'Location Sign\n(Information on Public Institutions etc.)',
    'H': 'Location Sign\n(Information on Service Facilities etc.)',
    'I': 'Location Sign\n(Information on Interesting Destinations etc.)',
    'J': 'Information Sign',
    'P': 'Signals by Policemen',
    'S': 'Symbols',
    'T': 'Additional Board',
    'X': 'Other',
    'Z': 'Uncategorized',
}
LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'P', 'S', 'T', 'X', 'Z']


def write_annotations(file_dir: str, out_dir: str) -> None:
    """
    Write annotations to a .csv-file
    :param file_dir: str, dir to the image files
    :param out_dir: str, .csv-file to write to
    :return: None
    """
    with open(out_dir, 'w') as out_file:
        for i in range(len(LETTERS)):
            for file in listdir(file_dir):
                if isfile(join(file_dir, file)) and LETTERS[i] in file:
                    out_file.write(f'{file},{LETTERS[i]}\n')


def rename_images(file_dir: str, out_dir: str) -> None:
    """
    Rename image files
    :param file_dir: str, dir to the image files
    :param out_dir: str, new dir
    :return: None
    """
    for i in range(len(LETTERS)):
        for j, file in enumerate(listdir(file_dir)):
            if LETTERS[i] in file:
                rename(f'{file_dir}{file}', f'{out_dir}{LETTERS[i]}_0{j+1}.jpg')


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

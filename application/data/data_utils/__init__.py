import cv2
import numpy as np
import pandas as pd
import skimage.io

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


def write_annotations(in_file: str, out_file: str, img_dir: str) -> None:
    """
    Write annotations to .csv-file based on image names in the images' directory (img_dir)
    :param in_file: str, file to get annotations from
    :param out_file: str, file to write relevant annotations to
    :param img_dir: str, directory with the images' files
    :return: None
    """
    import os

    with open(in_file, 'r') as in_f:
        with open(out_file, 'w') as out_f:
            out_f.write('img_name,category,id,sign,description\n')
            for line in in_f.readlines():
                if line.split(',')[0] in os.listdir(img_dir):
                    out_f.write(line)


def get_images(dataset: pd.DataFrame, key: str, img_dir: str) -> list[skimage.io.imread]:
    """
    Get images from image path
    :param dataset: pandas.Dataframe, dataset to get image names from
    :param key: str, key to dataset image names
    :param img_dir: str, image directory in project
    :return: list, list of the images
    """
    from skimage.io import imread
    from skimage.transform import resize
    from tqdm import tqdm

    temp = []

    for img_name in tqdm(dataset[key]):
        image_path = img_dir + str(img_name)
        img = imread(image_path, as_gray=True)
        img = resize(img, (100, 100), anti_aliasing=True)
        img /= 255
        img = img.astype('float32')
        temp.append(img)

    return temp


def binary_encoder(training_y: list[str], validation_y: list[str]) -> tuple[bin, bin, list[str], list[str]]:
    """
    Encodes string labels to binary
    :param training_y: list, y labels for training data (str)
    :param validation_y: list, y labels för validation data (str)
    :return: tuple of lists, training_y and validation_y in binary form
                                and copies of the original training_y and validation_y
    """
    from sklearn.preprocessing import LabelBinarizer

    encoder = LabelBinarizer()
    temp_train = training_y.copy()
    training_y = encoder.fit_transform(training_y)
    temp_val = validation_y.copy()
    validation_y = encoder.fit_transform(validation_y)

    return training_y, validation_y, temp_train, temp_val


def print_labels(binary_labels: bin, original_labels: list[str]) -> None:
    """Simple print function to print the binary value for the y labels"""
    for i, label in enumerate(original_labels):
        print(f'{binary_labels[i]} = {label}')


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

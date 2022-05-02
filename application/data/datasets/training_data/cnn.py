import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD


class TimerError(Exception):
    """Custom exception for timer error"""


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        import time

        if self._start_time is not None:
            raise TimerError(f'Timer is !\nUse .stop() to stop it.')

        self._start_time = time.perf_counter()

    def stop(self):
        import time

        if self._start_time is None:
            raise TimerError(f'Timer is not running!\n Use .start() to start it.')

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f'Elapsed time: {elapsed_time:.4f} seconds')


class Net(Module):
    def __init__(self, size: int) -> None:
        super(Net, self).__init__()
        self.cnn_layers = Sequential(
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_layers = Sequential(
            Linear(2500, size)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x


def write_annotations(in_file: str, out_file: str, img_dir: str) -> None:
    """
    Write annotations to .csv-file based on image names in the images' directory (img_dir)
    :param in_file: str, file to get annotations from
    :param out_file: str, file to write relevant annotations to
    :param img_dir: str, directory with the images' files
    :return: None
    """
    with open(in_file, 'r') as in_f:
        with open(out_file, 'w') as out_f:
            out_f.write('img_name,category,id,sign,description\n')
            for line in in_f.readlines():
                if line.split(',')[0] in os.listdir(img_dir):
                    out_f.write(line)


def train_model(epoch: int, model: Net, train_x, train_y, test_x, test_y) -> None:

    optimizer = Adam(model.parameters(), lr=0.07)
    criterion = CrossEntropyLoss()
    train_losses, val_losses = [], []

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    model.train()
    tr_loss = 0

    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()

    optimizer.zero_grad()

    output_train = model(train_x)
    output_val = model(test_x)

    loss_train = criterion(output_train, train_y)
    loss_val = criterion(output_val, test_y)
    train_losses.append(loss_train)
    val_losses.append(loss_val)

    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()

    if epoch % 2 == 0:
        print(f'Epoch: {epoch + 1}\tLoss: {loss_val}')


def get_images(dataset: pd.DataFrame, key: str, img_dir: str) -> list[skimage.io.imread]:
    """
    Get images from image path
    :param dataset: pandas.Dataframe, dataset to get image names from
    :param key: str, key to dataset image names
    :param img_dir: str, image directory in project
    :return: list, list of the images
    """
    from skimage.io import imread

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


def main():
    # in_annotations = './wikipedia/wikipedia_annotations.csv'
    out_annotations = '../wikipedia_annotations.csv'
    images_directory = './wikipedia/'

    timer = Timer()
    # write_annotations(in_annotations, out_annotations, images_directory)

    train_data = pd.read_csv(out_annotations)

    images = get_images(
        dataset=train_data,
        key='img_name',
        img_dir=images_directory
    )

    train_x = np.array(images)
    train_y = train_data['category']

    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1)

    train_x = train_x.reshape(train_x.shape[0], 1, 100, 100)
    train_x = torch.from_numpy(train_x)

    test_x = test_x.reshape(test_x.shape[0], 1, 100, 100)
    test_x = torch.from_numpy(test_x)

    model = Net(train_x.shape[0])

    train_y, test_y, training_labels, validation_labels = binary_encoder(train_y, test_y)
    # print_labels(train_y, training_labels)
    # print_labels(val_y, validation_labels)

    train_y = train_y.astype(int)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)
    train_y = train_y.resize_(train_y.shape[0],)

    test_y = test_y.astype(int)
    test_y = torch.from_numpy(test_y).type(torch.LongTensor)
    test_y = test_y.resize_(test_y.shape[0],)

    timer.start()

    for epoch in range(100):
        train_model(epoch, model, train_x, train_y, test_x, test_y)

    with torch.no_grad():
        output_train = model(train_x.cuda()) if torch.cuda.is_available() else model(train_x)
        output_test = model(test_x.cuda()) if torch.cuda.is_available() else model(test_x)

    softmax = torch.exp(output_train).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    print(f'Train accuracy score: {accuracy_score(train_y, predictions)}')

    softmax = torch.exp(output_test).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    print(f'Test accuracy score: {accuracy_score(test_y, predictions)}')

    timer.stop()


if __name__ == '__main__':
    main()

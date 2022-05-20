import glob as glob
import numpy as np
import os
import random as r
import torch

from PIL import Image
from torch.utils.data import Dataset
from xml.etree import ElementTree as et
from torchvision.transforms import transforms

from application.src.config import TEST_PATH
from application.src.models.faster_rcnn import RCNNModel
from application.src.utils.presentation import get_relevant_scores, show

from application.src.config import TRAIN_DIR, RESIZE_TO, CLASSES
from application.src.utils.transform import get_train_transform
from application.src.utils.presentation import show


class DS(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None) -> None:
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes

        self.image_paths = glob.glob(f'{self.dir_path}/*.jpg')
        self.all_images = [image_path.split('\\')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, i: int) -> tuple:
        image_name = self.all_images[i]
        image_path = os.path.join(self.dir_path, image_name)

        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        image = Image.open(image_path)
        image_width, image_height = image.size[1], image.size[0]

        boxes, labels = [], []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))

            # left corner x-coordinates
            x_min = int(member.find('bndbox').find('xmin').text)
            # right corner x-coordinates
            x_max = int(member.find('bndbox').find('xmax').text)
            # left corner y-coordinates
            y_min = int(member.find('bndbox').find('ymin').text)
            # right corner y-coordinates
            y_max = int(member.find('bndbox').find('ymax').text)

            x_min = (x_min / image_width) * self.width
            x_max = (x_max / image_width) * self.width
            y_min = (y_min / image_height) * self.height
            y_max = (y_max / image_height) * self.height

            boxes.append([x_min, y_min, x_max, y_max])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd,
            'image_id': torch.tensor([i])
        }

        if self.transforms:
            sample = self.transforms(
                image=np.array(image),
                bboxes=target['boxes'],
                labels=labels
            )
            image = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        image = image.swapaxes(0, 1)
        image = image.swapaxes(1, 2)

        return image, target

    def __len__(self) -> int:
        return len(self.all_images)


# prepare datasets and data loaders
train_dataset = DS(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())


def load_model():
    from application.src.config import DEVICE, MODEL_NAME, OUT_DIR

    model_path = str(OUT_DIR) + '/' + str(MODEL_NAME) + '.pth'

    nm = RCNNModel()
    nm.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    nm.model.eval()

    return nm


def print_result(labels: list, scores=None) -> None:
    """
    Print the labels and, if we predict a random image, the accuracy score.
    Only prints the signs that with more than 60% accuracy
    :param labels: list; list with the predicted or actual signs
    :param scores: list[int]; list with accuracy scores, only when predicted. None by default
    :return: None
    """
    if scores:
        [print(f'Sign: {label} - Accuracy: {scores[i]}%') for i, label in enumerate(labels)]
    else:
        [print(f'Sign: {label}') for label in labels]


def main():
    model = load_model()
    transform = transforms.Compose([transforms.ToTensor()])
    filename = r.choice(os.listdir(TEST_PATH))

    if filename[-3:] == 'jpg':
        path = '%s/%s' % (TEST_PATH, filename)
    else:
        filename = filename[:-3] + 'jpg'
        path = '%s/%s' % (TEST_PATH, filename)

    image = Image.open(path)

    image = transform(image)
    pred = image.view(1, 3, image.shape[1], image.shape[2])

    labels, scores, boxes = model.predict(pred)

    labels, scores, boxes = get_relevant_scores(labels, scores, boxes)

    image = image.swapaxes(0, 1)
    image = image.swapaxes(1, 2)

    show(image, labels, boxes)
    print_result(labels, scores)
    #
    # image, _ = train_dataset[10]
    #
    # show(image, [], [])


if __name__ == '__main__':
    main()

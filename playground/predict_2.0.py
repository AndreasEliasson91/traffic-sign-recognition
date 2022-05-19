import matplotlib.pyplot as plt
import numpy as np
import os
import random as r
import torch

from PIL import Image
from torchvision.transforms import transforms

from application.src.config import CLASSES
from application.src.models.faster_rcnn import RCNNModel
from application.src.utils import get_root_path

PATH = str(get_root_path()) + '/data/datasets/play_data'

LOWER_ACCURACY_LIMIT = 60


def _get_relevant_scores(labels, scores, boxes):
    x = len([score for score in scores if score >= LOWER_ACCURACY_LIMIT])
    return labels[:x], scores[:x], boxes[:x]


def show(image, labels, boxes):
    from matplotlib.patches import Rectangle

    plt.imshow(np.transpose(image[0].numpy(), (1, 2, 0)))

    for i in range(len(boxes)):
        x_min = int(boxes[i][0])
        y_min = int(boxes[i][1])
        x_max = int(boxes[i][2])
        y_max = int(boxes[i][3])

        plt.gca().add_patch(Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            edgecolor='red',
            facecolor=None,
            fill=False,
            lw=1
        ))

        plt.text(x_min, y_min - 10, str(labels[i]), fontsize=6, color='r')
    plt.show()


def load_model():
    from application.src.config import DEVICE, NUM_CLASSES, MODEL_NAME, OUT_DIR

    model_path = str(OUT_DIR) + '/' + str(MODEL_NAME) + '.pth'

    nm = RCNNModel()
    nm.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    nm.model.eval()

    return nm


def test(model, image, target) -> tuple[list, list, list[int], list]:
    with torch.no_grad():
        predictions = model(image)[0]

        correct_labels = [CLASSES[label] for label in target['labels']]
        predicted_labels = [CLASSES[label] for label in predictions['labels']]
        scores = [int(score * 100) for score in predictions['scores']]
        boxes = [box for box in predictions['boxes']]

    return correct_labels, predicted_labels, scores, boxes


def predict(model, image) -> tuple[list, list[int], list]:
    with torch.no_grad():
        predictions = model(image)[0]

        predicted_labels = [CLASSES[label] for label in predictions['labels']]
        scores = [int(score * 100) for score in predictions['scores']]
        boxes = [box for box in predictions['boxes']]

    return predicted_labels, scores, boxes


def print_result(labels: list, scores=None) -> None:
    """
    Print the labels and, if we predict a random image, the accuracy score.
    Only prints the signs that with more than 60% accuracy
    :param labels: list; list with the predicted or actual signs
    :param scores: list[int]; list with accuracy scores, only when predicted. None by default
    :return: None
    """
    if scores:
        print('*' * 20, '\n\nPredicted signs:')
        [print(f'Sign: {label} - Accuracy: {scores[i]}%') for i, label in enumerate(labels)]
    else:
        print('*' * 20, '\n\nActual signs:')
        [print(f'Sign: {label}') for label in labels]


if __name__ == '__main__':
    model = load_model()
    transform = transforms.Compose([transforms.ToTensor()])
    filename = r.choice(os.listdir(PATH))

    if filename[-3:] == 'jpg':
        path = '%s/%s' % (PATH, filename)
    else:
        filename = filename[:-3] + 'jpg'
        path = '%s/%s' % (PATH, filename)

    image = Image.open(path)
    image = transform(image)
    image = image.view(1, 3, image.shape[1], image.shape[2])

    labels, scores, boxes = model.predict(image)

    labels, scores, boxes = _get_relevant_scores(labels, scores, boxes)

    show(image, labels, boxes)
    print_result(labels, scores)

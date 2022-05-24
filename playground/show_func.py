import os
import random as r
import torch

from PIL import Image
from torchvision.transforms import transforms

from application.src.config import TEST_PATH
from application.src.models.faster_rcnn import load_model
from application.src.utils.presentation import get_relevant_scores, show

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
    image = image.view(1, 3, image.shape[1], image.shape[2])

    labels, scores, boxes = model.predict(image)

    labels, scores, boxes = get_relevant_scores(labels, scores, boxes)

    show(image, labels, boxes)


if __name__ == '__main__':
    main()

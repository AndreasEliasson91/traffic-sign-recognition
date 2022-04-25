import os

import cv2
import numpy as np
import matplotlib.image as mpimg

from application.data.data_utils import stack_images, write_annotations
from application.data.models import CustomImageDataset
from os import listdir, rename
from torchvision import transforms

wikipedia_dataset = CustomImageDataset(
    annotations_file='../application/data/datasets/training_data/wikipedia_signs/annotation/annotations_wikipedia_data.csv',
    img_dir='../application/data/datasets/training_data/wikipedia_signs/'
)


def main():
    # wikipedia_dataset.transform = transforms.ToPILImage()

    # wikipedia_dataset.plot_images(rows=10, cols=7)

    image_path = os.path.join('../application/data/datasets', 'training_data')
    print(image_path)


if __name__ == '__main__':
    main()


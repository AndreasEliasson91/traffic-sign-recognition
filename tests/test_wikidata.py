import cv2
import numpy as np

from application.data.datasets.training_data import wikipedia_dataset
from application.data.data_utils import stack_images
from os import listdir
from torchvision import transforms


def main():
    print(wikipedia_dataset[0])
    print(wikipedia_dataset[76])

    wikipedia_dataset.transform = transforms.ToPILImage()
    wikipedia_dataset.plot_images(index=76)
    wikipedia_dataset.plot_images(cols=3, rows=3)

    wikipedia_dataset.plot_images(rows=3, cols=3)
    imgs = []
    for i, image in enumerate(listdir(wikipedia_dataset.img_dir)):
        # print(image)
        img = cv2.imread(f'{wikipedia_dataset.img_dir}/{image}')
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.Canny(img, 100, 200)
            print(img.shape)
            imgs.append(img)

    stack = stack_images(.3, (imgs))
    cv2.imshow('Stack', stack)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()

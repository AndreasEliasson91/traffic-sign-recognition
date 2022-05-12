import matplotlib.pyplot as plt
import os
import pandas as pd
import torch

from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file: str, img_dir: str, transform=None, target_transform=None) -> None:
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, i: int) -> tuple:
        """
        Loads and returns a sample from the datasets at a given index.
        :param i: int, index
        :return: tuple
        """
        from torchvision.io import read_image

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[i, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[i, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def plot_images(self, cols=1, rows=1, index=None) -> None:
        """
        Plot images in a given size (format: cols x rows)
        :param cols: int, number of columns
        :param rows: int, number of rows
        :param index: int, get image by index else all
        :return: None
        """
        from application.src.config import WIKIPEDIA_LABELS

        figure = plt.figure(figsize=(8, 8))

        if not index:
            for i in range(1, cols * rows + 1):
                sample = torch.randint(len(self), size=(1,)).item()
                img, label = self[sample]
                figure.add_subplot(rows, cols, i)
                # plt.title(LABELS[label])
                plt.axis("off")
                plt.imshow(img)
        else:
            img, label = self[index]
            plt.title(WIKIPEDIA_LABELS[label])
            plt.axis("off")
            plt.imshow(img)

        plt.show()

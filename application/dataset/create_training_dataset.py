import sys

from application.dataset import CustomImageDataset
from torchvision import transforms

PROJECT_ROOT = sys.path[1]


def main():
    training_dataset = CustomImageDataset(
        annotations_file=f'{PROJECT_ROOT}/data/training_data/signs/annotation/training_annotations.csv',
        img_dir=f'{PROJECT_ROOT}/data/training_data/signs/'
    )

    print(training_dataset[0])

    training_dataset.transform = transforms.ToPILImage()
    # training_dataset.plot_images(index=76)
    training_dataset.plot_images(cols=3, rows=3)

    # training_dataset.plot_images(rows=3, cols=3)


if __name__ == '__main__':
    main()

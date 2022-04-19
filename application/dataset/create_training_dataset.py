import sys

from application.dataset import CustomImageDataset
from torchvision import transforms

PROJECT_ROOT = sys.path[1]
LABELS = {
    'A': 'Warning Sign',
    'B': 'Priority Sign',
    'C': 'Prohibitory Sign',
    'D': 'Mandatory Sign',
    'E': 'Instruction Sign',
    'F': 'Location Sign (Directions)',
    'G': 'Location Sign (Information on Public Institutions etc.)',
    'H': 'Location Sign (Information on Service Facilities etc.)',
    'I': 'Location Sign (Information on Interesting Destinations etc.)',
    'J': 'Information Sign',
    'P': 'Signals by Policemen',
    'S': 'Symbols',
    'T': 'Additional Board',
    'X': 'Other',
    'Z': 'Uncategorized',
}


def plot_images(dataset: CustomImageDataset) -> None:
    import matplotlib.pyplot as plt
    import torch

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(LABELS[label])
        plt.axis("off")
        plt.imshow(img)
    plt.show()


def main():
    training_dataset = CustomImageDataset(
        annotations_file=f'{PROJECT_ROOT}/data/training_data/signs/annotation/training_annotations.csv',
        img_dir=f'{PROJECT_ROOT}/data/training_data/signs/'
    )

    # Print the first sample of the dataset as a tensor
    print(training_dataset[0])

    # Convert dataset to PIL Images and plot 9 random images
    training_dataset.transform = transforms.ToPILImage()
    plot_images(training_dataset)


if __name__ == '__main__':
    main()

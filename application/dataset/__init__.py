import sys

from application.dataset.dataset import CustomImageDataset
from torchvision import transforms

PROJECT_ROOT = sys.path[1]
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

wikipedia_dataset = CustomImageDataset(
    annotations_file=f'{PROJECT_ROOT}/data/training_data/signs/annotation/training_annotations.csv',
    img_dir=f'{PROJECT_ROOT}/data/training_data/signs/',
    transform=transforms.ToPILImage()
)

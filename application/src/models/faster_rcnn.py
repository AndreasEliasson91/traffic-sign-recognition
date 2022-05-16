import glob as glob
import numpy as np
import os
import torch

from application.src.config import TRAIN_DIR, RESIZE_TO, VALID_DIR, CLASSES, BATCH_SIZE
from application.src.utils.transform import get_train_transform, get_valid_transform
from application.src.utils import collate_fn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from xml.etree import ElementTree as et


class LUDataset(Dataset):
    """Creating a dataset to use in the RCNN-model"""
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
        image_path = os.path.join(self.dir_path, self.all_images[i])
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype(np.float32)

        image_resized = image_arr.resize((self.width, self.height))

        annot_filename = image_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)

        boxes, labels = [], []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        image_width = image.shape[1]
        image_height = image.shape[0]

        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))

            # left corner x-coordinates
            xmin = int(member.find('bndbox').find('xmin').text)
            # right corner x-coordinates
            xmax = int(member.find('bndbox').find('xmax').text)
            # left corner y-coordinates
            ymin = int(member.find('bndbox').find('ymin').text)
            # right corner y-coordinates
            ymax = int(member.find('bndbox').find('ymax').text)

            xmin_final = (xmin / image_width) * self.width
            xmax_final = (xmax / image_width) * self.width
            ymin_final = (ymin / image_height) * self.height
            yamx_final = (ymax / image_height) * self.height

            boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

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
                image=image_resized,
                bboxes=target['boxes'],
                labels=labels
            )
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self) -> int:
        return len(self.all_images)


# prepare datasets and data loaders
train_dataset = LUDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
valid_dataset = LUDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)
print(f'Number of training samples: {len(train_dataset)}')
print(f'Number of validation samples: {len(valid_dataset)}\n')


def create_model(num_classes: int):
    """Initialize model and predictor"""
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

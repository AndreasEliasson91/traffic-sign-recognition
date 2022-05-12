import albumentations
import albumentations as A
import cv2
import numpy as np

from albumentations.pytorch import ToTensorV2
from application.src.config import DEVICE


def show_transformed_image(train_loader) -> None:
    """Debug tool, to show the transformed images"""
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))

            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()

            for box in boxes:
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255), 2)
            cv2.imshow('sample', sample)


def get_train_transform() -> albumentations.Compose:
    return A.Compose([ToTensorV2(p=1.0)], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def get_valid_transform() -> albumentations.Compose:
    return A.Compose([ToTensorV2(p=1.0)], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

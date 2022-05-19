
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from application.src.config import CLASSES, DEVICE, NUM_CLASSES


class RCNNModel:
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        self.model.to(DEVICE)

    def verify(self, image, target) -> tuple[list, list, list[int], list]:
        with torch.no_grad():
            predictions = self.model(image)[0]

            correct_labels = [CLASSES[label] for label in target['labels']]
            predicted_labels = [CLASSES[label] for label in predictions['labels']]
            scores = [int(score * 100) for score in predictions['scores']]
            boxes = [box for box in predictions['boxes']]

        return correct_labels, predicted_labels, scores, boxes

    def predict(self, image) -> tuple[list, list[int], list]:
        with torch.no_grad():
            predictions = self.model(image)[0]

            predicted_labels = [CLASSES[label] for label in predictions['labels']]
            scores = [int(score * 100) for score in predictions['scores']]
            boxes = [box for box in predictions['boxes']]

        return predicted_labels, scores, boxes

import cv2
import torch
import numpy as np

from application.src.utils import get_root_path
from application.src.utils.transform import get_train_transform
from application.src.models.faster_rcnn import CustomDataset, create_model
from application.src.config import RESIZE_TO, CLASSES

PATH = str(get_root_path()) + '/data/datasets/LU-data/valid'
DATASET = CustomDataset(PATH, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())

GET_ALL_PREDICTIONS = True


def predict(model, image, target):
    predictions = model(image)

    # print(predictions[0]['boxes'], '\n\n', target['boxes'])
    if target:
        actual_labels = [CLASSES[label] for label in target['labels']]

    predicted_labels = [CLASSES[prediction] for prediction in predictions[0]['labels']]
    scores = [int(score * 100) for score in predictions[0]['scores']]

    if target and not GET_ALL_PREDICTIONS:
        x = len(actual_labels)
        predicted_labels = predicted_labels[:x]
        scores = scores[:x]

    return actual_labels, predicted_labels, scores


def load_model():
    from application.src.config import DEVICE, NUM_CLASSES, MODEL_NAME, OUT_DIR

    path = str(OUT_DIR) + '/' + str(MODEL_NAME) + '.pth'
    model = create_model(NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()

    return model


def main():
    with torch.no_grad():
        index = len(DATASET) - 1
        item = DATASET[index]

        image = item[0].unsqueeze_(0)
        target = item[1]

        print(image.shape)

        image_path = str(get_root_path()) + '/data/datasets/play_data/a.jpg'

        image = cv2.imread(image_path)
        image = torch.from_numpy(image)
        image = torch.flip(image, [-2])

        print(image.shape)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (RESIZE_TO, RESIZE_TO))
        image_resized /= 255.0

        model = load_model()
        labels, predictions, scores = predict(model, image, None)

        print('*' * 20, '\n\nActual signs:')
        [print(f'Sign: {label}') for label in labels]

        print('*' * 20, '\n\nPredicted signs:')
        [print(f'Sign: {prediction} - Accuracy: {scores[i]}%') for i, prediction in enumerate(predictions)]


if __name__ == '__main__':
    main()

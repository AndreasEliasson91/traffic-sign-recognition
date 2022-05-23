import matplotlib.pyplot as plt
import io

from PIL import Image


def get_relevant_scores(labels: list, scores: list, boxes: list) -> tuple[list, list, list]:
    """
    Get score over a set lower accuracy limit
    :param labels: list; predicted labels
    :param scores: list; predicted accuracy scores
    :param boxes: list; predicted bounding boxes
    :return: tuple[list, list, list];
    """
    from application.src.config import LOWER_ACCURACY_LIMIT

    x = len([score for score in scores if score >= LOWER_ACCURACY_LIMIT])
    return labels[:x], scores[:x], boxes[:x]


def show(image, labels, boxes) -> None:
    """
    Plot and show the image with the models predicted bounding boxes and labels
    :param image: PIL image obj; input image
    :param labels: list; predicted labels
    :param boxes: list; predicted bounding boxes
    :return: None
    """
    from matplotlib.patches import Rectangle

    plt.imshow(image)

    for i in range(len(boxes)):
        x_min = int(boxes[i][0])
        y_min = int(boxes[i][1])
        x_max = int(boxes[i][2])
        y_max = int(boxes[i][3])

        plt.gca().add_patch(Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            edgecolor='red',
            facecolor=None,
            fill=False,
            lw=1
        ))

        plt.text(x_min, y_min - 10, str(labels[i]), fontsize=6, color='r')
    plt.show()


def convert_upload(upload):
    file = upload
    file_name = str(list(file.keys())[0])
    bytes_image = file[file_name]['content']
    PIL_image = Image.open(io.BytesIO(bytes_image))
    return PIL_image


def print_result(labels: list, scores=None) -> None:
    """
    Print the labels and, if we predict a random image, the accuracy score.
    Only prints the signs that with more than 60% accuracy
    :param labels: list; list with the predicted or actual signs
    :param scores: list[int]; list with accuracy scores, only when predicted. None by default
    :return: None
    """
    if scores:
        [print(f'Sign: {label} - Accuracy: {scores[i]}%') for i, label in enumerate(labels)]
    else:
        [print(f'Sign: {label}') for label in labels]

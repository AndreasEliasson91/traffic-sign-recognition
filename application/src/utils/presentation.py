import matplotlib.pyplot as plt
import io

from PIL import Image


COLORS = [
    '#d90166',
    '#8f00f1',
    '#d0ff14',
    '#eb5030',
    '#ff000d',
    '#66ff00',
    '#0203e2',
    '#04d9ff',
    '#ff00ff',
    '#fffd01',
    '#e56024',
    '#dfff4f',
    '#ff3503',
    '#6600ff',
    '#f7b718',
    '#fe0002',
    '#45cea2',
    '#ff85ff',
    '#1974d2',
    '#fe6700',
]


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


def show(image: Image, labels: list, boxes: list, scores: list) -> None:
    """
    Plot and show the image with the models predicted bounding boxes and labels
    :param image: PIL image obj; input image
    :param labels: list; predicted labels
    :param boxes: list; predicted bounding boxes
    :param scores: list; prediction scores
    :return: None
    """
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    patches = []
    fig, ax = plt.subplots(figsize=(13, 7))
    plt.imshow(image)

    for i in range(len(boxes)):
        label = f'{str(labels[i])} - {str(scores[i])}%'

        x_min = int(boxes[i][0])
        y_min = int(boxes[i][1])
        x_max = int(boxes[i][2])
        y_max = int(boxes[i][3])

        plt.gca().add_patch(Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            edgecolor=COLORS[i],
            facecolor=None,
            fill=False,
            lw=1
        ))

        patch = Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=COLORS[i],
            label=label
        )
        patches.append(patch)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        handles=[patch for patch in patches]
    )
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

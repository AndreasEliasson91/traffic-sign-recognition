import matplotlib.pyplot as plt
import numpy as np


def get_relevant_scores(labels, scores, boxes):
    from application.src.config import LOWER_ACCURACY_LIMIT

    x = len([score for score in scores if score >= LOWER_ACCURACY_LIMIT])
    return labels[:x], scores[:x], boxes[:x]


def show(image, labels, boxes):
    from matplotlib.patches import Rectangle

    plt.imshow(np.transpose(image[0].numpy(), (1, 2, 0)))

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
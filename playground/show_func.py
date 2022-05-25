import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms import transforms

from application.src.models.faster_rcnn import load_model
from application.src.utils.presentation import get_relevant_scores


colors = [
    '#d90166',
    '#8f00f1',
    '#0203e2',
    '#d0ff14',
    '#eb5030',
    '#ff000d',
    '#66ff00',
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


def show(image, labels, boxes, scores) -> None:
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
            edgecolor=colors[i],
            facecolor=None,
            fill=False,
            lw=1
        ))

        patch = Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], label=label)
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


def main():
    model = load_model()
    transform = transforms.Compose([transforms.ToTensor()])

    image = Image.open('../application/data/datasets/play_data/b.jpg')
    prediction_image = transform(image)
    prediction_image = prediction_image.view(1, 3, prediction_image.shape[1], prediction_image.shape[2])

    labels, scores, boxes = model.predict(prediction_image)
    # labels, scores, boxes = get_relevant_scores(labels, scores, boxes)

    show(image, labels, boxes, scores)


if __name__ == '__main__':
    main()

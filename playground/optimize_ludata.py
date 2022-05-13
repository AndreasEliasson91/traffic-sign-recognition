from matplotlib import pyplot as plt

from application.src.models.faster_rcnn import train_dataset, create_model, train_loader
from application.src.config import DEVICE
from application.src.utils import Averager
from tqdm import tqdm

import torch

from application.src.utils.timer import Timer


def train(train_data_loader, model, train_loss_hist) -> list[int]:
    print('Training the model')

    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    train_itr, val_itr = 1, 1
    train_loss_list, val_loss_list = [], []

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1

        prog_bar.set_description(desc=f'Loss: {loss_value:.4f}')

    return train_loss_list


def main():
    model = create_model(num_classes=23)
    model = model.to(DEVICE)

    train_loss_hist = Averager()
    val_loss_hist = Averager()

    timer = Timer()

    for epoch in range(1):
        print('None Transform')
        train_loss_hist.reset()
        val_loss_hist.reset()

        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()

        timer.start()

        train_loss = train(train_loader, model, train_loss_hist)
        print(f'Epoch #{epoch} train loss: {train_loss_hist.value:.3f}')
        print(f'Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}')

        timer.stop()


if __name__ == '__main__':
    main()

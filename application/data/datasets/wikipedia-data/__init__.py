import numpy as np
import pandas as pd
import torch

from application.src.utils import Timer
from application.data.data_utils import binary_encoder, get_images
from application.src.models import NeuralNetwork
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


def main():
    # in_annotations = './wikipedia/wikipedia_annotations.csv'
    out_annotations = './wikipedia_annotations.csv'
    images_directory = './wikipedia/'

    timer = Timer()
    # write_annotations(in_annotations, out_annotations, images_directory)

    train_data = pd.read_csv(out_annotations)

    images = get_images(
        dataset=train_data,
        key='img_name',
        img_dir=images_directory
    )

    train_x = np.array(images)
    train_y = train_data['category']

    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1)

    train_x = train_x.reshape(train_x.shape[0], 1, 100, 100)
    train_x = torch.from_numpy(train_x)

    test_x = test_x.reshape(test_x.shape[0], 1, 100, 100)
    test_x = torch.from_numpy(test_x)

    model = NeuralNetwork(train_x.shape[0])

    train_y, test_y, training_labels, validation_labels = binary_encoder(train_y, test_y)
    # print_labels(train_y, training_labels)
    # print_labels(val_y, validation_labels)

    train_y = train_y.astype(int)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)
    train_y = train_y.resize_(train_y.shape[0],)

    test_y = test_y.astype(int)
    test_y = torch.from_numpy(test_y).type(torch.LongTensor)
    test_y = test_y.resize_(test_y.shape[0],)

    optimizer = Adam(model.parameters(), lr=0.07)
    criterion = CrossEntropyLoss()

    timer.start()

    for epoch in range(100):
        model.train_nn(
            epoch=epoch,
            optimizer=optimizer,
            criterion=criterion,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y
        )

    with torch.no_grad():
        output_train = model(train_x.cuda()) if torch.cuda.is_available() else model(train_x)
        output_test = model(test_x.cuda()) if torch.cuda.is_available() else model(test_x)

    softmax = torch.exp(output_train).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    print(f'Train accuracy score: {accuracy_score(train_y, predictions)}')

    softmax = torch.exp(output_test).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)

    print(f'Test accuracy score: {accuracy_score(test_y, predictions)}')

    timer.stop()


if __name__ == '__main__':
    main()

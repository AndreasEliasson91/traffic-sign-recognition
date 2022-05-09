import numpy as np
import torch
import torch.nn as nn
import hiddenlayer as hl


def main():
    x_train = np.array([[1.7], [2.5], [5.5], [7.9], [8.8], [2.4], [2.4], [8.89], [5], [4.4]], dtype=np.float32)
    y_train = np.array([[1.9], [2.68], [4.22], [8.19], [9.69], [3.4], [2.6], [8.8], [5.6], [4.7]], dtype=np.float32)
    X_train = torch.tensor(x_train)
    Y_train = torch.tensor(y_train)

    print(X_train.shape)

    inp = 1
    out = 1
    hid = 100

    model1 = torch.nn.Sequential(torch.nn.Linear(inp, hid), torch.nn.Linear(hid, out))
    print(hl.build_graph(model1, torch.zeros([10, 1])))

if __name__ == '__main__':
    main()

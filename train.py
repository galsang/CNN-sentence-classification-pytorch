from .model import CNN

import torch.optim as optim
import torch.nn as nn

from torchtext import data
from torchtext import datasets


def train(lr=0.1):
    model = CNN()

    optimizer = optim.Adadelta(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    TEXT = data.Field

    optimizer.zero_grad()
    output = CNN(input)
    loss = criterion(output, target)
    loss.backword()
    optimizer.step()

    return model


def main():
    train()


if __name__ == "__main__":
    main()

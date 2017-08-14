from model import CNN

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
from sklearn.utils import shuffle


def train(train_x, train_y, dev_x, dev_y, params):
    word_idx = params["word_idx"]
    classes = params["classes"]

    model = CNN(**params).cuda()

    optimizer = optim.Adadelta(model.parameters(), params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    epoch = 0
    pre_dev_acc = 0
    while(True):
        epoch += 1

        for i in range(0, len(train_x), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(train_x) - i)

            batch_x = Variable(torch.LongTensor(
                [[word_idx[w] for w in sent] + [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                 for sent in train_x[i:i + batch_range]])).cuda()
            batch_y = Variable(torch.LongTensor([classes.index(c) for c in train_y[i:i + batch_range]])).cuda()

            optimizer.zero_grad()
            pred = model(batch_x, is_train=True)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            # constrain l2-norms of the weight vectors
            if model.fc.weight.norm() > params["NORM_LIMIT"]:
                model.fc.weight.data = model.fc.weight.data * params["NORM_LIMIT"] / model.fc.weight.data.norm()

        dev_acc = test(dev_x, dev_y, model, params)
        print("epoch:", epoch, ", acc:", dev_acc)
        if dev_acc <= pre_dev_acc:
            print("early stopping!")
            break
        else:
            pre_dev_acc = dev_acc

    return model


def test(test_x, test_y, model, params):
    vocab = params["vocab"]
    word_idx = params["word_idx"]
    classes = params["classes"]

    batch_x = []
    for sent in test_x:
        sent_idx = []
        for w in sent:
            if w in vocab:
                sent_idx.append(word_idx[w])
            else:
                sent_idx.append(params["VOCAB_SIZE"])

        sent_idx += [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
        batch_x.append(sent_idx)

    batch_x = Variable(torch.LongTensor(batch_x)).cuda()
    batch_y = [classes.index(c) for c in test_y]

    pred = np.argmax(model(batch_x).cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, batch_y)]) / len(pred)

    return acc



def read_data(path):
    x, y = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            y.append(line[:-1].split()[0].split(":")[0])
            x.append(line[:-1].split()[1:])

    return shuffle(x, y)


def main():
    train_x, train_y = read_data("data/TREC/TREC_train.txt")
    dev_idx = len(train_x)//10
    dev_x, dev_y, train_x, train_y = train_x[:dev_idx], train_y[:dev_idx], train_x[dev_idx:], train_y[dev_idx:]
    test_x, test_y = read_data("data/TREC/TREC_test.txt")

    vocab = sorted(list(set([w for sent in train_x for w in sent])))
    classes = sorted(list(set(train_y)))
    word_idx = {w: i for i, w in enumerate(vocab)}

    params = {
        "vocab": vocab,
        "classes": classes,
        "word_idx": word_idx,
        "BATCH_SIZE": 50,
        "MAX_SENT_LEN": max([len(sent) for sent in train_x]),
        "WORD_DIM": 300,
        "IN_CHANNEL": 1,
        "VOCAB_SIZE": len(vocab),
        "CLASS_SIZE": len(classes),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "LEARNING_RATE": 0.1
    }

    model = train(train_x, train_y, dev_x, dev_y, params)
    test_acc = test(test_x, test_y, model, params)
    print("test acc:", test_acc)


if __name__ == "__main__":
    for i in range(10):
        main()

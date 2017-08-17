from model import CNN
import utils

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
import numpy as np
import argparse


def train(data, params):
    model = CNN(**params).cuda()

    optimizer = optim.Adadelta(model.parameters(), params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    pre_dev_acc = 0
    dev_count = 0
    for e in range(params["EPOCH"]):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

            batch_x = [[data["word_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]
            batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]

            batch_x = Variable(torch.LongTensor(batch_x)).cuda()
            batch_y = Variable(torch.LongTensor(batch_y)).cuda()

            optimizer.zero_grad()
            pred = model(batch_x, is_train=True)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            # constrain l2-norms of the weight vectors
            if model.fc.weight.norm() > params["NORM_LIMIT"]:
                model.fc.weight.data = model.fc.weight.data * params["NORM_LIMIT"] / model.fc.weight.data.norm()

        dev_acc = test(data, model, params, mode="dev")
        print("epoch:", e+1, "/ dev_acc:", dev_acc)

        if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
            dev_count += 1
            if dev_count >= 3:
                print("early stopping!")
                break
        else:
            pre_dev_acc = dev_acc

    return model


def test(data, model, params, mode="test"):
    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]

    x = [[data["word_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

    x = Variable(torch.LongTensor(x)).cuda()
    y = [data["classes"].index(c) for c in y]

    pred = np.argmax(model(x).cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)

    return acc


def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train(with test) a model / test: test saved models")
    parser.add_argument("--model", default="rand", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default="TREC", help="available datasets: MR, TREC")
    parser.add_argument("--save_model", default="F", help="whether saving model or not (T/F)")
    parser.add_argument("--early_stopping", default="F", help="whether to apply early stopping(T/F)")
    parser.add_argument("--epoch", default=10, type=int, help="number of max epoch")

    options = parser.parse_args()
    data = getattr(utils, f"read_{options.dataset}")()

    # ToDO: use test data for builing vocab?
    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["dev_x"] + data["test_x"] for w in sent])))
    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_idx"] = {w: i for i, w in enumerate(data["vocab"])}

    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "SAVE_MODEL": bool(options.save_model == "T"),
        "EARLY_STOPPING": bool(options.early_stopping == "T"),
        "EPOCH": options.epoch,
        # ToDO: use test data for estimating MAX_SENT_LEN?
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
        "BATCH_SIZE": 50,
        "WORD_DIM": 300,
        "IN_CHANNEL": 1,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "LEARNING_RATE": 0.01
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    print("DATASET:", params["DATASET"])
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])
    print("=" * 20 + "INFORMATION" + "=" * 20)

    if options.mode == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        model = train(data, params)
        if params["SAVE_MODEL"]:
            utils.save_model(model, params)
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)

    else:
        model = utils.load_model(params).cuda()

    test_acc = test(data, model, params)
    print("test acc:", test_acc)

    return test_acc


if __name__ == "__main__":
    print("max test acc:", max([main() for i in range(10)]))

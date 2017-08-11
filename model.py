import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.BATCH_SIZE = 50
        self.MAX_SENT_LEN = 100
        self.WORD_DIM = 300
        self.IN_CHANNEL = 1
        self.VOCAB_SIZE = 50000

        self.FILTERS = [3, 4, 5]
        self.FILTER_NUM = [100, 100, 100]
        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        self.convs = [
            nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            for i in range(len(self.FILTERS))
        ]

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.VOCAB_SIZE)

    def forward(self):
        x = Variable(torch.randn(self.BATCH_SIZE, self.IN_CHANNEL, self.WORD_DIM, self.MAX_SENT_LEN))
        x = x.view(-1, self.IN_CHANNEL, self.WORD_DIM * self.MAX_SENT_LEN)

        # (N, C_out, MAX_SENT_LEN - FILTER_LEN + 1)

        conv_results = [
            F.max_pool1d(
                F.relu(self.convs[i](x)),
                self.MAX_SENT_LEN - self.FILTERS[i] + 1)
                .view(-1, self.FILTER_NUM[i])
            for i in range(len(self.FILTERS))
        ]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=0.5, training=False)
        x = self.fc(x)

        return x


import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
import json

from aggregator import *
from encoder import *


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.criterion = torch.nn.L1Loss(reduction = "mean")
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
#         print(embeds)
#         print("EMBEDS: ")
#         print(embeds.shape)
        scores = self.weight.mm(embeds)
#         print("SCORES: ")
#         print(scores.shape)
#         print("\n")
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        self.scores = scores
        return self.criterion(scores, labels)
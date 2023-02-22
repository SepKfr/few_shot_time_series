import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PoswiseFeedForwardNet(nn.Module):

    def __init__(self, d_model, d_ff, seed):
        super(PoswiseFeedForwardNet, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, inputs):

        return self.w_2(F.gelu(self.w_1(inputs)))
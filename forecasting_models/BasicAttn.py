import numpy as np
import torch
import torch.nn as nn
import random

from modules.clustering import Clustering


class BasicAttn(nn.Module):

    def __init__(self, d_k, h, device, seed, few_shot):

        super(BasicAttn, self).__init__()

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.device = device
        self.d_k = d_k

        self.few_shot = few_shot
        self.few_shot = few_shot
        if self.few_shot:
            self.clustering = Clustering(device=device, d_model=d_k * h)
            self.layer_norm = nn.LayerNorm(d_k, elementwise_affine=False, device=device)
            self.w1 = nn.Sequential(nn.Linear(d_k, d_k, device=self.device),
                                    nn.GELU())

    def forward(self, Q, K, V, attn_mask):

        if self.few_shot:

            context_clustering, loss = self.clustering(Q, K, V)
            scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(self.d_k)
            attn = torch.softmax(scores, -1)
            context = torch.einsum('bhqk,bhkd->bhqd', attn, V)

            context_final = self.layer_norm(context + self.w1(context_clustering))

            return context_final, attn, loss

        else:
            scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(self.d_k)
            attn = torch.softmax(scores, -1)
            context = torch.einsum('bhqk,bhkd->bhqd', attn, V)
            return context, attn
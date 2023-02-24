import numpy as np
import torch.nn as nn
import torch


class Clustering(nn.Module):
    def __init__(self, *, device, num_clusters=3, d_model):
        super(Clustering, self).__init__()

        self.device = device
        self.num_clusters = num_clusters

        self.proj_back_to_cluster_k = nn.Sequential(nn.Linear(num_clusters, d_model, device=self.device),
                                                    nn.GELU())

        self.cluster_k_proj = nn.Sequential(nn.Linear(d_model, num_clusters, device=self.device),
                                            nn.GELU())

        self.cluster_q_proj = nn.Sequential(nn.Linear(d_model, num_clusters, device=self.device),
                                            nn.GELU())

        self.cluster_v_proj = nn.Sequential(nn.Linear(d_model, num_clusters, device=self.device),
                                            nn.GELU())

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, Q, K, V):

        b, h, l, d_k = Q.shape
        l_k = K.shape[2]

        unfolding = 10 * self.num_clusters

        padding = torch.zeros(unfolding, h, l_k, d_k, device=self.device)
        K_padded = torch.cat([padding, K[1:]])
        V_padded = torch.cat([padding, V[1:]])
        K_unfold = K_padded.unfold(0, unfolding, 1)
        V_unfold = V_padded.unfold(0, unfolding, 1)

        K_unfold = K_unfold.reshape(b, l_k, -1, d_k*h)
        Q_unfold = Q.reshape(b, l, -1, d_k*h)
        V_unfold = V_unfold.reshape(b, l, -1, d_k*h)

        cluster_k = self.cluster_k_proj(K_unfold)
        cluster_q = self.cluster_q_proj(Q_unfold)
        cluster_v = self.cluster_v_proj(V_unfold)

        scores = torch.einsum('blcd, blpd -> blcp', cluster_q, cluster_k) / np.sqrt(d_k*h)
        scores = torch.softmax(scores, -1)
        cluster_context = torch.einsum('blcp, blpd -> blcd', scores, cluster_v)

        mu = torch.mean(cluster_q, dim=-1)

        loss = - torch.mean(torch.logsumexp(scores, dim=-1)) + 0.001 * self.cross_entropy(mu, mu)

        cluster_center = self.proj_back_to_cluster_k(cluster_context).reshape(b, h, l_k, d_k)

        return cluster_center, loss

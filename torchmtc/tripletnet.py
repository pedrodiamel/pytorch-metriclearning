import torch
import torch.nn as nn
import torch.nn.functional as F

class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, anchor, positive, negative):
        embedded_a = self.embeddingnet(anchor)
        embedded_p = self.embeddingnet(positive)
        embedded_n = self.embeddingnet(negative)
        return embedded_a, embedded_p, embedded_n



"""
Taken from https://github.com/PeterWang512/GenDataAttribution/blob/main/networks.py
"""

import torch


class LinearProbe(torch.nn.Module):
    def __init__(self, model, feat_dim):
        super().__init__()
        self.model = model
        self.feat_dim = feat_dim

        for param in self.model.parameters():
            param.requires_grad = False

        self.mapper = torch.nn.Linear(self.feat_dim, self.feat_dim)
        self.mapper.weight.data.copy_(torch.eye(self.feat_dim))
        self.mapper.bias.data.fill_(0)

    def forward(self, images):
        x = self.model(images).float().detach()
        x = self.mapper(x)
        x = torch.nn.functional.normalize(x, dim=-1)
        return x

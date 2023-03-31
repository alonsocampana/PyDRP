import torch
from torch import nn
from torch_geometric import nn as gnn
from torch.nn import functional as F

class FCBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 p_dropout,
                 **kwargs):
        super().__init__()
        self.nw = nn.Sequential(nn.ReLU(),
                               nn.Dropout(p=p_dropout),
                               nn.Linear(input_dim, hidden_dim),
                               nn.Sigmoid(),
                               nn.Linear(hidden_dim, 1))
    def forward(self, x):
        return self.nw(x)

class AttnDropout(nn.Module):
    def __init__(self,
                 p_dropout = 0.1,
                 **kwargs):
        super().__init__()
        self.id = nn.Identity()
        self.bern = torch.distributions.bernoulli.Bernoulli(torch.Tensor([p_dropout]))
    def forward(self, x):
        x = self.id(x).squeeze()
        if self.training:
            mask = self.bern.sample([x.shape[0]]).squeeze()
            incorrect = mask.all().any()
            while incorrect:
                mask = self.bern.sample([x.shape[0]]).squeeze()
                incorrect = mask.all().any()
            x[mask.bool()] = float("-inf")
        return x

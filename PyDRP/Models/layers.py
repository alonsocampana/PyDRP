import torch
from torch import nn
from torch_geometric import nn as gnn
from torch.nn import functional as F
from PyDRP.Models.utils import init_weights
import torch_geometric 

class FCBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim = 1,
                 p_dropout = 0.0,
                 use_batchnorm = False,
                 **kwargs):
        super().__init__()
        if use_batchnorm:
            self.nw = nn.Sequential(nn.ReLU(),
                                   nn.BatchNorm1d(input_dim),
                                   nn.Dropout(p=p_dropout),
                                   nn.Linear(input_dim, hidden_dim),
                                   nn.Sigmoid(),
                                   nn.Linear(hidden_dim, output_dim))
        else:
            self.nw = nn.Sequential(nn.ReLU(),
                                   nn.Dropout(p=p_dropout),
                                   nn.Linear(input_dim, hidden_dim),
                                   nn.Sigmoid(),
                                   nn.Linear(hidden_dim, output_dim))
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
class ResNetGated(nn.Module):
    def __init__(self, init_dim, hidden_dim, layers, p_dropout):
        super().__init__()
        self.p_dropout = p_dropout
        assert layers > 0
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(init_dim, hidden_dim),
                             nn.ReLU(),
                             nn.Dropout(p=p_dropout),
                             nn.Linear( hidden_dim, init_dim)) for i in range(layers)])
        self.gates = nn.Parameter(torch.Tensor(layers))
        self.layers.apply(init_weights)
    def forward(self, x):
        range_gates = torch.sigmoid(self.gates)
        for i, layer in enumerate(self.layers):
            x = (range_gates[i])*layer(F.relu(x)) + (1-range_gates[i])*x
        return x

class GatedGNNRes(nn.Module):
    def __init__(
        
        self,
        layer,
        layer_kwargs,
        n_layers = 4,
        **kwargs
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([layer(**layer_kwargs) for i in range(n_layers)])
        self.gates = nn.Parameter(torch.zeros(n_layers))
    def forward(self, x, edge_index,edge_weight = None, batch=None):
        gates = torch.sigmoid(self.gates)
        for i in range(self.n_layers):
            x = (1 - gates[i]) * self.layers[i](F.leaky_relu(x), edge_index, edge_weight, batch) + ((gates[i]) * x)
        return x

class TransGAT(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 edge_dim,
                 num_heads,
                 share_kqv = False,
                 gat_dropout = 0.0,
                 n_gat_heads = 1,
                 skip=False,
                 **kwargs):
        super().__init__()
        self.skip = skip
        self.g_c = gnn.global_max_pool
        self.share_kqv = share_kqv
        if self.share_kqv:
            self.k = self.v = self.q= gnn.GATv2Conv(input_dim, output_dim, concat=False, edge_dim = edge_dim, dropout= gat_dropout)
        else:
            self.k = gnn.GATv2Conv(input_dim*2, output_dim, concat=False, edge_dim = edge_dim, dropout= gat_dropout, heads = n_gat_heads )
            self.v = gnn.GATv2Conv(input_dim, output_dim, concat=False, edge_dim = edge_dim, dropout= gat_dropout , heads = n_gat_heads)
            self.q = gnn.GATv2Conv(input_dim*2, output_dim, concat=False, edge_dim = edge_dim, dropout= gat_dropout , heads = n_gat_heads)
        self.pool = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        self.apply(init_weights)
        self.tdb = tdb = torch_geometric.utils.to_dense_batch
        
    def forward(self,x, edge_index, edge_attr, batch = None,*args, **kwargs):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if self.skip:
            x, mask = self.tdb(self.v(x, edge_index, edge_attr), batch)
            return x[mask].squeeze()
        if self.share_kqv:
            out = self.tdb(self.q(x, edge_index, edge_attr), batch)
            query = keys = values= out[0]
            mask = out[1]
        else:
            global_context = self.g_c(x, batch)
            x_ = torch.cat([x, torch.repeat_interleave(global_context, torch.bincount(batch), 0)], axis=-1)
            query = self.tdb(self.q(x_, edge_index, edge_attr), batch)[0]
            keys = self.tdb(self.k(x_, edge_index, edge_attr), batch)[0]
            values, mask = self.tdb(self.v(x, edge_index, edge_attr), batch)
        x, _ = self.pool(query, keys, values, key_padding_mask = ~mask)
        batch = out = query = keys = values = global_context = x_ = ""
        return x[mask].squeeze()

class MultiPooling(nn.Module):
    def __init__(self, input_dim, p_dropout=0.0):
        super().__init__()
        self.lin = nn.Linear(input_dim * 3, input_dim)
        self.maxp = gnn.global_max_pool
        self.meanp = gnn.global_mean_pool
        self.do = nn.Dropout(p = p_dropout)
    def forward(self, x, batch, *args, **kwargs):
        return self.lin(self.do(torch.cat([self.maxp(x, batch),
                                   - self.maxp(- x, batch),
                                   self.meanp(x, batch)], axis=1)))
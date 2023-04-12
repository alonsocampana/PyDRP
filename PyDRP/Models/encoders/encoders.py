import torch
from torch import nn
from torch_geometric import nn as gnn
from torch.nn import functional as F
from PyDRP.Models.PairsNetwork import CellEncoder, DrugEncoder, init_weights
from PyDRP.Models.layers import GatedGNNRes, TransGAT
from PyDRP.Models.NNlayers import AttnDropout

class GeneExpEncoder(CellEncoder):
    def __init__(self,
                 init_dim,
                 bottleneck_dim,
                 embed_dim,
                 activation = "relu",
                 genes_do = 0.0,
                 **kwargs):
        """
        GeneExpression Encoder. activation either relu, sigmoid, leaky_relu or elu
        """
        super().__init__()
        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "sigmoid":
            activation_fn = nn.Sigmoid
        elif activation == "leaky_relu":
            activation_fn = nn.LeakyReLU
        elif activation == "elu":
            activation_fn = nn.ELU
        self.do = nn.Dropout(genes_do)
        self.net = nn.Sequential(nn.Linear(init_dim, bottleneck_dim),
                                activation_fn(),
                                nn.Linear(bottleneck_dim, embed_dim))
    def forward(self, line_features, *args, **kwargs):
        return self.net(self.do(line_features))
    
class GATmannEncoder(DrugEncoder):
    def __init__(self,
                 node_features=79,
                 edge_features=10,
                 n_conv_layers = 3,
                 n_heads=1,
                 p_dropout_gat = 0.001,
                 embed_dim=32,
                 **kwargs):
        super().__init__()
        self.n_conv_layers = n_conv_layers
        self.gat_init = gnn.GATv2Conv(node_features, embed_dim, heads= n_heads, edge_dim=edge_features, dropout=p_dropout_gat)
        self.gat_layers = nn.ModuleList([gnn.GATv2Conv(embed_dim*n_heads,
                                             embed_dim*n_heads,
                                             heads= n_heads,
                                             edge_dim=edge_features,
                                             concat=False, dropout=p_dropout_gat) for g in range(n_conv_layers-1)])
        self.gat_init.apply(init_weights)
        [layer.apply(init_weights) for layer in self.gat_layers]
    def forward(self, x, edge_index, edge_attr, *args, **kwargs):
        x = self.gat_init(x, edge_index, edge_attr)
        for gat_layer in self.gat_layers:
            x = gat_layer(F.leaky_relu(x), edge_index, edge_attr)
        return x

class GTEncoder(DrugEncoder):
    def __init__(self, embed_dim, num_heads=1, n_layers=1):
        super().__init__()
        self.init_gat = gnn.GATv2Conv(79, embed_dim, edge_dim = 10)
        self.layers = _stack = GatedGNNRes(TransGAT,
                                           {"input_dim":embed_dim,
                                             "output_dim":embed_dim,
                                             "edge_dim":10,
                                             "num_heads":num_heads,},
                                           n_layers = n_layers)
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.init_gat(x, edge_index, edge_attr)
        return self.layers(x, edge_index, edge_attr, batch)

class GNNAttnDrugPooling(nn.Module):
    def __init__(self,
                 embed_dim,
                 hidden_dim,
                 output_embed_dim,
                 p_dropout_attn = 0.0,
                 p_dropout_nodes = 0.0,
                 **kwargs):
        super().__init__()
        self.pool = gnn.GlobalAttention(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                             nn.ReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(hidden_dim, 1),
                                                             AttnDropout(p_dropout_nodes)),
                                               nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                             nn.ReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(hidden_dim, output_embed_dim)))
    def set_cold(self):
        for p in self.parameters():
            p.requires_grad=False
    def forward(self, x, batch):
        return self.pool(x, batch)

class GNNMultiheadAttnDrugPooling(nn.Module):
    def __init__(self,
                 embed_dim,
                 hidden_dim,
                 output_embed_dim,
                 p_dropout_attn = 0.0,
                 p_dropout_nodes = 0.0,
                 p_dropout_output = 0.0,
                 n_heads = 4,
                 **kwargs):
        super().__init__()
        self.n_heads = n_heads
        self.pool = nn.ModuleList([gnn.GlobalAttention(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                             nn.ReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(hidden_dim, 1),
                                                             AttnDropout(p_dropout_nodes)),
                                               nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                             nn.ReLU(),
                                                             nn.Dropout(p_dropout_attn),
                                                             nn.Linear(hidden_dim, output_embed_dim),
                                                             nn.Dropout(p_dropout_output),)) for i in range(n_heads)])
    def set_cold(self):
        for p in self.parameters():
            p.requires_grad=False
    def forward(self, x, batch):
        x_ = self.pool[0](x, batch)
        for i in range(1, self.n_heads):
            x_ = x_ + self.pool[i](x, batch)
        return x_/self.n_heads
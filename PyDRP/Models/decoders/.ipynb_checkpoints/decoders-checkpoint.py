import torch
from torch import nn
from torch_geometric import nn as gnn
from torch.nn import functional as F
from PyDRP.Models.PairsNetwork import ResponseDecoder

class DotproductDecoder(ResponseDecoder):
    def __init__(self,
                 **kwargs):
        """
        Decoder via a simple dot product
        """
        super().__init__()
            
    def forward(self, line_embeds, drug_embeds, *args, **kwargs):
        return line_embeds.mul(drug_embeds).sum(axis=-1)

class FCDecoder(ResponseDecoder):
    def __init__(self,
                 init_dim,
                 hidden_dim,
                 output_dim=1,
                 p_dropout_1 = 0.0,
                 p_dropout_2 = 0.0,
                 **kwargs):
        """
        Decoder via concatenation and an MLP
        """
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(p_dropout_1),
                                nn.Linear(init_dim, hidden_dim),
                                nn.Sigmoid(),
                                nn.Dropout(p_dropout_2),
                                nn.Linear(hidden_dim, output_dim))
            
    def forward(self, line_embeds, drug_embeds, *args, **kwargs):
        return self.net(torch.cat([line_embeds, drug_embeds], axis=-1))
class NonlinearDotDecoder(ResponseDecoder):
    def __init__(self,
                 init_dim,
                 hidden_dim,
                 embed_dim,
                 p_dropout_1 = 0.0,
                 p_dropout_2 = 0.0,
                 **kwargs):
        """
        Decoder via mapping to latent features and dot product
        """
        super().__init__()
        self.net_1 = nn.Sequential(nn.Dropout(p_dropout_1),
                                nn.Linear(init_dim, hidden_dim),
                                nn.Sigmoid(),
                                nn.Dropout(p_dropout_2),
                                nn.Linear(hidden_dim, embed_dim))
        self.net_2 = nn.Sequential(nn.Dropout(p_dropout_1),
                                nn.Linear(init_dim, hidden_dim),
                                nn.Sigmoid(),
                                nn.Dropout(p_dropout_2),
                                nn.Linear(hidden_dim, embed_dim))
            
    def forward(self, line_embeds, drug_embeds, *args, **kwargs):
        return self.net_1(line_embeds).mul(self.net_2(drug_embeds)).sum(axis=-1)

class ModularDecoder(ResponseDecoder):
    def __init__(self,
                 init_dim,
                 hidden_dim_interaction,
                 p_dropout_interaction,
                 hidden_dim_drugs,
                 p_dropout_drugs,
                 hidden_dim_lines,
                 p_dropout_lines,
                 **kwargs):
        """
        Decoder via latent scores and addition
        """
        super().__init__()
        self.interaction_module = FCBlock(init_dim, hidden_dim_interaction, p_dropout_interaction)
        self.drug_module = FCBlock(init_dim, hidden_dim_drug, p_dropout_drug)
        self.line_module = FCBlock(init_dim, hidden_dim_line, p_dropout_line)
            
    def forward(self, line_embeds, drug_embeds, interaction_embeds, *args, **kwargs):
        return self.interaction_module(interaction_embeds) + self.drug_module(drug_embeds) + self.line_module(line_embeds)
import torch
from torch import nn
from torch_geometric import nn as gnn
from torch.nn import functional as F
from PyDRP.Models.layers import FCBlock

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)

class PairsNetwork(nn.Module):
    def __init__(self,
                 **kwargs):
        """
        Interface for a network taking as input lines and drugs and returning a scalar prediction.
        """
        super().__init__()
    def set_cold(self):
        for p in self.parameters():
            p.requires_grad=False
    def set_warm(self):
        for p in self.parameters():
            p.requires_grad=True
            
    def forward(self,  *args, **kwargs):
        raise NotImplementedError

class DoubleEncoderDecoder(PairsNetwork):
    def __init__(self,
                 line_encoder,
                 drug_encoder,
                 line_adapter,
                 drug_adapter,
                 decoder,
                 **kwargs):
        """
        Network consisting of two encoders, two adapters and a decoder.
        The forward method has to be reimplemented.
        """
        super().__init__()
        self.line_encoder = line_encoder
        self.drug_encoder = drug_encoder
        self.line_adapter = line_adapter
        self.drug_adapter = drug_adapter
        self.decoder = decoder
    def forward(self, line_f, drug_f, *args, **kwargs):
        raise NotImplementedError

class ModularEncoderDecoder(PairsNetwork):
    def __init__(self,
                 line_encoder,
                 drug_encoder,
                 line_adapter,
                 drug_adapter,
                 decoder,
                 **kwargs):
        """
        Interface consisting of two encoders, two adapters and a modular decoder. 
        The forward method has to be reimplemented.
        """
        super().__init__()
        self.line_encoder = line_encoder
        self.drug_encoder = drug_encoder
        self.line_adapter = line_adapter
        self.drug_adapter = drug_adapter
        self.interaction_adapter = interaction_adapter
        self.decoder = decoder
    def forward(self, line_f, drug_f, *args, **kwargs):
        raise NotImplementedError

class CellEncoder(nn.Module):
    def __init__(self,
                 **kwargs):
        """
        Interface for a gene encoder, it should take as input some line features, and return the encoded gene expression
        """
        super().__init__()
    def set_cold(self):
        for p in self.parameters():
            p.requires_grad=False
    def set_warm(self):
        for p in self.parameters():
            p.requires_grad=True
            
    def forward(self, line_features, *args, **kwargs):
        raise NotImplementedError

class DrugEncoder(nn.Module):
    def __init__(self,
                 **kwargs):
        """
        Interface for a drug encoder, it should take as input some drug features, and return the embeddings
        """
        super().__init__()
    def set_cold(self):
        for p in self.parameters():
            p.requires_grad=False
    def set_warm(self):
        for p in self.parameters():
            p.requires_grad=True
            
    def forward(self, line_features, *args, **kwargs):
        raise NotImplementedError

class ResponseDecoder(nn.Module):
    def __init__(self,
                 **kwargs):
        """
        Interface for a decoder, it should take as input some drug and line embeddings, and return a prediction
        """
        super().__init__()
    def set_cold(self):
        for p in self.parameters():
            p.requires_grad=False
    def set_warm(self):
        for p in self.parameters():
            p.requires_grad=True
            
    def forward(self, line_embeds, drug_embeds, *args, **kwargs):
        raise NotImplementedError
 
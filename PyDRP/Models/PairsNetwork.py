import torch
from torch import nn
from torch_geometric import nn as gnn
from torch.nn import functional as F
from PyDRP.Models.layers import FCBlock
from PyDRP.Models.utils import init_weights

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


class GNNProteinDrugEncoderDecoder(PairsNetwork):
    def __init__(self,
                 protein_encoder,
                 drug_encoder,
                 protein_adapter,
                 drug_adapter,
                 decoder,
                 **kwargs):
        """
        Network consisting of two encoders, two adapters and a decoder.
        """
        super().__init__()
        self.protein_encoder = protein_encoder
        self.drug_encoder = drug_encoder
        self.protein_adapter = protein_adapter
        self.drug_adapter = drug_adapter
        self.decoder = decoder
    def forward(self, data, *args, **kwargs):
        x_lines = self.protein_adapter(self.protein_encoder(data["protein"]))
        x_drugs = self.drug_adapter(self.drug_encoder(data["x"],
                                                      data["edge_index"],
                                                      data["edge_attr"],
                                                      data["batch"]),
                                    data["batch"])
        return self.decoder(x_lines, x_drugs)

class GNNCellDrugEncoderDecoder(PairsNetwork):
    def __init__(self,
                 line_encoder,
                 drug_encoder,
                 line_adapter,
                 drug_adapter,
                 decoder,
                 **kwargs):
        """
        Network consisting of two encoders, two adapters and a decoder.
        """
        super().__init__()
        self.line_encoder = line_encoder
        self.drug_encoder = drug_encoder
        self.line_adapter = line_adapter
        self.drug_adapter = drug_adapter
        self.decoder = decoder
    def forward(self, data, *args, **kwargs):
        x_lines = self.line_adapter(self.line_encoder(data["cell"]))
        x_drugs = self.drug_adapter(self.drug_encoder(data["x"],
                                                      data["edge_index"],
                                                      data["edge_attr"],
                                                      data["batch"]),
                                    data["batch"])
        return self.decoder(x_lines, x_drugs)

class GNNDrugEncoderDecoder(PairsNetwork):
    def __init__(self,
                 drug_encoder,
                 drug_adapter,
                 decoder,
                 **kwargs):
        """
        Network consisting of two encoders, two adapters and a decoder.
        """
        super().__init__()
        self.drug_encoder = drug_encoder
        self.drug_adapter = drug_adapter
        self.decoder = decoder
        self.multitask = False
    def make_multitask(self, n_tasks, module, module_kwargs, share_init_w = True):
        self.decoder = nn.ModuleList([])
        for n in range(n_tasks):
            new_head = module(**module_kwargs)
            self.decoder += [new_head]
        if share_init_w:
            for n in range(n_tasks):
                if n == 0:
                    state_dict = self.decoder[n].state_dict()
                else:
                    self.decoder[n].load_state_dict(state_dict)
        self.multitask = True
        self.current_task = None
    def set_task(self, n_task):
        self.current_task = n_task
    def forward(self, data, *args, **kwargs):
        x_drugs = self.drug_adapter(self.drug_encoder(data["x"],
                                                      data["edge_index"],
                                                      data["edge_attr"],
                                                      data["batch"]),
                                    data["batch"])
        if self.multitask:
            return self.decoder[self.current_task](x_drugs)
        else:
            return self.decoder(x_drugs)
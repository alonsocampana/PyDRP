import torchmetrics
import numpy as np
import torch
from torchmetrics import Metric

class ElementwiseMetric(Metric):
    """
    Computes a metric averaged over the categories determined by average, using the functional metric defined by aggr_fn.
    average:string [either "drugs" or "lines"]. Default: "drugs"
    aggr_fn: torchmetrics.functional function, or a function computing the desired metric.
    
    Methods:
    update: updates the metric
    compute: computes the metric
    get_dict: Gets the metric for each element in the level
    """
    full_state_update = True
    def __init__(self, average ="drugs", aggr_fn = torchmetrics.functional.pearson_corrcoef):
        super().__init__(full_state_update=True)
        self.add_state("drugs", default=[])
        self.add_state("lines", default=[])
        self.add_state("observed", default=[])
        self.add_state("predicted", default=[])
        self.average = average
        self.aggr_fn = aggr_fn

    def update(self, preds, target, drugs, lines):
        self.drugs += drugs
        self.lines += lines
        self.observed += [preds]
        self.predicted += [target]

    def _drugs_compute(self):
        fs = []
        obs = torch.cat(self.observed).squeeze()
        pred = torch.cat(self.predicted).squeeze()
        d_arry = np.array(self.drugs).squeeze()
        for d in np.unique(d_arry):
            mask = torch.Tensor(d_arry == d).bool().to(self.device)
            fs+= [self.aggr_fn(obs[mask], pred[mask])]
        return torch.stack(fs).mean()
    def _lines_compute(self):
        fs = []
        obs = torch.cat(self.observed).squeeze()
        pred = torch.cat(self.predicted).squeeze()
        l_arry = np.array(self.lines).squeeze()
        for l in np.unique(l_arry):
            mask = torch.Tensor(l_arry == l).bool().to(self.device)
            fs+= [self.aggr_fn(obs[mask], pred[mask])]
        return torch.stack(fs).nanmean()
    def _drugs_compute_nagg(self):
        fs = {}
        obs = torch.cat(self.observed).squeeze()
        pred = torch.cat(self.predicted).squeeze()
        d_arry = np.array(self.drugs).squeeze()
        for d in np.unique(d_arry):
            mask = torch.Tensor(d_arry == d).bool().to(self.device)
            fs[d] = self.aggr_fn(obs[mask], pred[mask])
        return torch.stack(fs).nanmean()
    def _lines_compute_nagg(self):
        fs = {}
        obs = torch.cat(self.observed).squeeze()
        pred = torch.cat(self.predicted).squeeze()
        l_arry = np.array(self.lines).squeeze()
        for l in np.unique(l_arry):
            mask = torch.Tensor(l_arry == l).bool().to(self.device)
            fs[l] = self.aggr_fn(obs[mask], pred[mask])
        return fs
    def compute(self):
        if self.average =="drugs":
            return self._drugs_compute()
        elif self.average =="lines":
            return self._lines_compute()
        else:
            raise NotImplementedError
    def get_dict(self):
        if self.average =="drugs":
            return self._drugs_compute_nagg()
        elif self.average =="lines":
            return self._lines_compute_nagg()
        else:
            raise NotImplementedError
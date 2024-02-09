from PyDRP.Data import DatasetManager, GDSC, PRISM, CTRPv2
from PyDRP.Data.features.drugs import GraphCreator
from PyDRP.Data.features.cells import TensorLineFeaturizer
from PyDRP.Data.features.targets import MinMaxScaling, IdentityPipeline
import pandas as pd
from pprint import pprint
import os
import numpy as np
from torch import nn
from torch_geometric import nn as gnn
import torch
import torch_geometric
import torchmetrics

import torch
from torch import Tensor
from torchmetrics import Metric
import torchmetrics

class GroupwiseMetric(Metric):
    def __init__(self, metric,
                 grouping = "cell_lines",
                 average = "macro",
                 nan_ignore=True,
                 alpha=0.00001,
                 residualize = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.grouping = grouping
        self.metric = metric
        self.average = average
        self.nan_ignore = nan_ignore
        self.residualize = residualize
        self.alpha = alpha
        self.add_state("target", default=torch.tensor([]))
        self.add_state("pred", default=torch.tensor([]))
        self.add_state("drugs", default=torch.tensor([]))
        self.add_state("cell_lines", default=torch.tensor([]))
    def get_residual(self, X, y):
        w = self.get_linear_weights(X, y)
        r = y-(X@w)
        return r
    def get_linear_weights(self, X, y):
        A = X.T@X
        Xy = X.T@y
        n_features = X.size(1)
        A.flatten()[:: n_features + 1] += self.alpha
        return (torch.linalg.solve(A, Xy).T).squeeze()
    def get_residual_ind(self, y, drug_id, cell_id, alpha=0.001):
        X = torch.cat([y.new_ones(y.size(0), 1), torch.nn.functional.one_hot(drug_id.squeeze()), torch.nn.functional.one_hot(cell_id.squeeze())], 1).float()
        return self.get_residual(X, y)

    def compute(self) -> Tensor:
        if self.grouping == "cell_lines":
            grouping = self.cell_lines
        elif self.grouping == "drugs":
            grouping = self.drugs
        metric = self.metric
        if not self.residualize:
            y_obs = self.target
            y_pred = self.pred
        else:
            y_obs = self.get_residual_ind(self.target, self.drugs, self.cell_lines)
            y_pred = self.get_residual_ind(self.pred, self.drugs, self.cell_lines)
        average = self.average
        nan_ignore = self.nan_ignore
        unique = grouping.unique()
        proportions = []
        metrics = []
        for g in unique:
            is_group = grouping == g
            metrics += [metric(y_obs[grouping == g], y_pred[grouping == g])]
            proportions += [is_group.sum()/len(is_group)]
        if average is None:
            return torch.stack(metrics)
        if (average == "macro") & (nan_ignore):
            return torch.nanmean(torch.Tensor([metrics]))
        if (average == "macro") & (not nan_ignore):
            return torch.mean(torch.Tensor([metrics]))
        if (average == "micro") & (not nan_ignore):
            return (torch.Tensor([proportions])*torch.Tensor([metrics])).sum()
        else:
            raise NotImplementedError
    
    def update(self, preds: Tensor, target: Tensor,  drugs: Tensor,  cell_lines: Tensor) -> None:
        self.target = torch.cat([self.target, target])
        self.pred = torch.cat([self.pred, preds])
        self.drugs = torch.cat([self.drugs, drugs]).long()
        self.cell_lines = torch.cat([self.cell_lines, cell_lines]).long()
        
def get_residual(X, y, alpha=0.001):
    w = get_linear_weights(X, y, alpha=alpha)
    r = y-(X@w)
    return r
def get_linear_weights(X, y, alpha=0.01):
    A = X.T@X
    Xy = X.T@y
    n_features = X.size(1)
    A.flatten()[:: n_features + 1] += alpha
    return torch.linalg.solve(A, Xy).T
def residual_correlation(y_pred, y_obs, drug_id, cell_id):
    X = torch.cat([torch.ones(y_pred.size(0), 1), torch.nn.functional.one_hot(drug_id), torch.nn.functional.one_hot(cell_id)], 1).float()
    r_pred = get_residual(X, y_pred)
    r_obs = get_residual(X, y_obs)
    return torchmetrics.functional.pearson_corrcoef(r_pred, r_obs)

def get_residual_ind(y, drug_id, cell_id, alpha=0.001):
    X = torch.cat([torch.ones(y.size(0), 1), torch.nn.functional.one_hot(drug_id), torch.nn.functional.one_hot(cell_id)], 1).float()
    return get_residual(X, y, alpha=alpha)

def average_over_group(y_obs, y_pred, metric, grouping, average="macro", nan_ignore = False):
    unique = grouping.unique()
    proportions = []
    metrics = []
    for g in unique:
        is_group = grouping == g
        metrics += [metric(y_obs[grouping == g], y_pred[grouping == g])]
        proportions += [is_group.sum()/len(is_group)]
    if average is None:
        return torch.stack(metrics)
    if (average == "macro") & (nan_ignore):
        return torch.nanmean(torch.Tensor([metrics]))
    if (average == "macro") & (not nan_ignore):
        return torch.mean(torch.Tensor([metrics]))
    if (average == "micro") & (not nan_ignore):
        return (torch.Tensor([proportions])*torch.Tensor([metrics])).sum()
    else:
        raise NotImplementedError

class CancerDataset():
    def __init__(self, inhibitions, drug_dict, cell_line_dict):
        self.inhibitions = inhibitions
        self.drug_dict = drug_dict
        self.line_dict = cell_line_dict
        self.drug_ids = {k:i for i, k in enumerate(self.drug_dict.keys())}
        self.line_ids = {k:i for i, k in enumerate(self.line_dict.keys())}
    def __len__(self):
        return self.inhibitions.shape[0]
    def __getitem__(self, idx):
        pair = self.inhibitions.iloc[[idx]]
        cell = pair.loc[:, "CELL_ID"].item()
        drug = pair.loc[:, "DRUG_ID"].item()
        y = pair.loc[:, ["Y"]]
        drug_data = self.drug_dict[drug].clone()
        cell_data = self.line_dict[cell].clone()
        drug_data["y"] = torch.Tensor(y.to_numpy())
        drug_data["cell_id"] = torch.Tensor([self.line_ids[cell]]).unsqueeze(0)
        drug_data["drug_id"] = torch.Tensor([self.drug_ids[drug]]).unsqueeze(0)
        drug_data["cell"] = cell_data.unsqueeze(0)
        return drug_data

class BenchCanc():
    def __init__(self,
                 config,
                 fold,
                 merge_train_eval = True,
                 setting = "precision_oncology",
                 dataset = "GDSC2",
                 line_features = "expression",
                 epoch_callback=None,
                 final_callback=None,):
        self.line_features = line_features
        self.seed = 3558
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.dataset = dataset
        self.setting = setting
        self.max_patience = 10
        self.n_folds = 10
        self.fold = fold
        self.config = config
        self.merge_train_eval = merge_train_eval
        if epoch_callback is None:
            self.epoch_callback = self.default_epoch_callback
        else:
            self.epoch_callback = epoch_callback
        if final_callback is None:
            self.final_callback = self.default_final_callback
        else:
            self.final_callback = final_callback
    def _get_manager(self):
        paccmann_genes = pd.read_csv("https://raw.githubusercontent.com/prassepaul/mlmed_ranking/main/data/gdsc_data/paccmann_gene_list.txt", index_col=None, header=None).to_numpy().squeeze().tolist()
        if self.setting == "precision_oncology":
            partition_column = "CELL_ID"
        else:
            partition_column = "DRUG_ID"
        if self.dataset == "GDSC1":
            manager = DatasetManager(processing_pipeline = GDSC(target = "LN_IC50",
                                                                gene_subset = paccmann_genes,
                                                                cell_lines = self.line_features),
                                    target_processor = IdentityPipeline(),
                                    partition_column = "DRUG_ID",
                                    k = self.n_folds,
                                    drug_featurizer = GraphCreator(),
                                    line_featurizer = TensorLineFeaturizer())
        if self.dataset == "GDSC2":
            manager = DatasetManager(processing_pipeline = GDSC(target = "LN_IC50",
                                                                dataset = "GDSC2",
                                                                gene_subset = paccmann_genes,
                                                                cell_lines = self.line_features),
                                    target_processor = IdentityPipeline(),
                                    partition_column = "DRUG_ID",
                                    k = self.n_folds,
                                    drug_featurizer = GraphCreator(),
                                    line_featurizer = TensorLineFeaturizer())
        if self.dataset == "CTRPv2":
            manager = DatasetManager(processing_pipeline = CTRPv2(target = "ec50",
                                                    gene_subset = paccmann_genes,
                                                    clip_val = 10,
                                                    cell_lines = self.line_features),
                        target_processor = IdentityPipeline(),
                        partition_column = "DRUG_ID",
                        k = self.n_folds,
                        drug_featurizer = GraphCreator(),
                        line_featurizer = TensorLineFeaturizer())
        if self.dataset == "PRISM":
            manager = DatasetManager(processing_pipeline = PRISM(target = "ec50",
                                                    gene_subset = paccmann_genes,
                                                    clip_val = 10,
                                                    cell_lines = self.line_features),
                        target_processor = IdentityPipeline(),
                        partition_column = "DRUG_ID",
                        k = 10,
                        drug_featurizer = GraphCreator(),
                        line_featurizer = TensorLineFeaturizer())
        return manager
    def _instantiate(self, model, fold):
        self.patience = self.max_patience + 0
        self.best_train_loss = 1000
        manager = self._get_manager()
        self.manager = manager
        drug_dict = self._get_drug_representation(manager)
        line_dict = self._get_cell_representation(manager)
        train, val, test = self._get_fold(manager, self.fold)
        if self.merge_train_eval:
            train = pd.concat([train, val], 0)
            val_dataloader = None
        else:
            val_dataloader = self._create_dataloader(val, drug_dict, line_dict, shuffle=False)
        train_dataloader = self._create_dataloader(train, drug_dict, line_dict, shuffle=True)
        test_dataloader = self._create_dataloader(test, drug_dict, line_dict, shuffle=False)
        self.device = torch.device(self.config["env"]["device"])
        optimizer = torch.optim.Adam(model.parameters(), self.config["optimizer"]["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               patience=self.config["optimizer"]["patience"],
                                                               factor=0.1)
        loss = nn.MSELoss()
        model.to(self.device)
        scaler = torch.cuda.amp.GradScaler()
        return optimizer, scheduler, scaler, loss, train_dataloader, val_dataloader, test_dataloader
    def train_model(self, model):
        scaler = torch.cuda.amp.GradScaler()
        optimizer, scheduler, scaler, loss, train_dataloader, val_dataloader, test_dataloader = self._instantiate(model, self.fold)
        train_metrics = self._get_train_metrics().to(self.device)
        self.train_metrics = train_metrics
        test_metrics = self._get_eval_metrics().to(self.device)
        self.test_metrics = test_metrics
        for epoch in range(self.config["optimizer"]["max_epochs"]):
            train_metrics.reset()
            test_metrics.reset()
            self.train_step(model, scaler, optimizer, loss, train_metrics, train_dataloader)
            self.eval_step(model, scaler, test_metrics, test_dataloader)
            self.epoch_callback(epoch, model, train_metrics, test_metrics)
            if self.early_stop(train_metrics["MeanSquaredError"]):
                break
        return self.final_callback(model, train_metrics, test_metrics)
    def train_step(self,
                   model,
                   scaler,
                   optimizer,
                   loss,
                   metrics,
                   dataloader):
        model.train()
        for batch in dataloader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=self.config["env"]["mixed_precision"]):
                y_pred = model(batch)
                l = loss(y_pred, batch["y"])
            scaler.scale(l).backward()
            if self.config["optimizer"]["clip_norm"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["optimizer"]["clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            metrics.update(y_pred.squeeze(),
                           batch["y"].squeeze().to(self.device),
                           drugs = batch["drug_id"].squeeze().to(self.device),
                           cell_lines = batch["cell_id"].squeeze().to(self.device))
    def eval_step(self,
                   model,
                   scaler,
                   metrics,
                   dataloader):
        model.eval()
        for batch in dataloader:
            with torch.no_grad():
                batch = batch.to(self.device)
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=self.config["env"]["mixed_precision"]):
                    y_pred = model(batch)
                metrics.update(y_pred.squeeze(),
                               batch["y"].squeeze().to(self.device),
                               drugs = batch["drug_id"].squeeze().to(self.device),
                               cell_lines = batch["cell_id"].squeeze().to(self.device))
    def default_epoch_callback(self, epoch, model, train_metrics, test_metrics):
        test_metrics = {it[0]:it[1].item() for it in test_metrics.compute().items()}
        print(f"epoch : {epoch}, test_metrics: {test_metrics}")
    def early_stop(self, last_train_loss):
        if last_train_loss < self.best_train_loss:
            self.best_train_loss = last_train_loss + 0
            self.patience = self.max_patience + 0
            return False
        else:
            self.patience -= 1
        if self.patience <= 0:
            return True
        else:
            return False
    def default_final_callback(self, model, train_metrics, test_metrics):
        return test_metrics.compute()
    def _get_train_metrics(self):
        if self.setting == "drug_discovery":
            return torchmetrics.MetricCollection({"R_drugwise_residuals":GroupwiseMetric(metric=torchmetrics.functional.pearson_corrcoef,
                              grouping="drugs",
                              average="macro",
                              residualize=True),
                    "R_drugwise":GroupwiseMetric(metric=torchmetrics.functional.pearson_corrcoef,
                                          grouping="drugs",
                                          average="macro",
                                          residualize=False),
                    "MeanSquaredError":torchmetrics.MeanSquaredError()})
        elif self.setting == "precision_oncology":
            return torchmetrics.MetricCollection({"R_cellwise_residuals":GroupwiseMetric(metric=torchmetrics.functional.pearson_corrcoef,
                              grouping="cell_lines",
                              average="macro",
                              residualize=True),
                    "R_cellwise":GroupwiseMetric(metric=torchmetrics.functional.pearson_corrcoef,
                                          grouping="cell_lines",
                                          average="macro",
                                          residualize=True),
                    "MeanSquaredError":torchmetrics.MeanSquaredError()})
    def _get_eval_metrics(self):
        if self.setting == "drug_discovery":
            return torchmetrics.MetricCollection({"R_drugwise_residuals":GroupwiseMetric(metric=torchmetrics.functional.pearson_corrcoef,
                              grouping="drugs",
                              average="macro",
                              residualize=True),
                    "R_drugwise":GroupwiseMetric(metric=torchmetrics.functional.pearson_corrcoef,
                                          grouping="drugs",
                                          average="macro",
                                          residualize=False),
                    "MeanSquaredError":torchmetrics.MeanSquaredError()})
        elif self.setting == "precision_oncology":
            return torchmetrics.MetricCollection({"R_cellwise_residuals":GroupwiseMetric(metric=torchmetrics.functional.pearson_corrcoef,
                              grouping="cell_lines",
                              average="macro",
                              residualize=True),
                    "R_cellwise":GroupwiseMetric(metric=torchmetrics.functional.pearson_corrcoef,
                                          grouping="cell_lines",
                                          average="macro",
                                          residualize=False),
                    "MeanSquaredError":torchmetrics.MeanSquaredError()})
    def _get_drug_representation(self, manager):
        return manager.get_drugs()
    def _get_cell_representation(self, manager):
        return manager.get_cell_lines()
    def _get_fold(self, manager, fold):
        return manager.get_partition(fold)
    def _create_dataloader(self, inhibitions, drug_dict, cell_dict, shuffle=False):
        return torch.utils.data.DataLoader(CancerDataset(inhibitions, drug_dict, cell_dict),
                                           collate_fn = torch_geometric.data.Batch.from_data_list,
                                           batch_size=self.config["optimizer"]["batch_size"],
                                           shuffle=shuffle, num_workers = 8)
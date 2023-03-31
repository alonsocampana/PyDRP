from PyDRP.src import DatasetManager
import pandas as pd
import numpy as np
import os
import torch

class FixedExperimentManager(DatasetManager):
    def __init__(self,
                 experiment_name,
                 drug_dict,
                 line_dict,
                 drug_featurizer,
                 line_featurizer,
                 list_dfs_train,
                 list_dfs_test,
                 list_dfs_val = None,
                 ):
        self.experiment_name = experiment_name,
        self.line_dict = line_dict
        self.drug_dict = drug_dict
        self.data_train = list_dfs_train
        self.data_test = list_dfs_test
        self.data_val = list_dfs_val
        self.drug_featurizer = drug_featurizer
        self.line_featurizer = line_featurizer
    def get_data(self):
        return pd.concat(self.data_test, ignore_index = True, axis=0)
    def get_partition(self, idx):
        train, test = self.data_train[idx], self.data_test[idx]
        if self.data_val is not None:
            val = self.data_val[idx]
        else:
            val = None
        return train, test, val
    def get_cell_lines(self):
        cell_lines = self.line_dict
        return self.line_featurizer(cell_lines.to_numpy(),
                                    cell_lines.index.to_numpy())
    def get_drugs(self):
        smiles =  self.drug_dict
        path_cache = f"data/processed/{self.experiment_name}.pt"
        if os.path.exists(path_cache):
            drug_dict = torch.load(path_cache)
        else:
            drug_dict = self.drug_featurizer(list(smiles.iloc[:, 0].to_numpy()), smiles.index.to_numpy())
            torch.save(drug_dict, path_cache)
        featurized_drugs = set(list(drug_dict.keys()))
        input_drugs = set(list(smiles.index.to_numpy()))
        diff_drugs = featurized_drugs.difference(input_drugs)
        self.missing_drugs = diff_drugs
        if len(diff_drugs) > 0:
            warnings.warn(f"it was not possible to featurize {diff_drugs}", RuntimeWarning)
        return drug_dict
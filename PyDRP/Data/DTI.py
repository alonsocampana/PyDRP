from tdc.multi_pred import DTI
import re
import torch
import numpy as np
from PyDRP.src import Splitter
import pandas as pd
import os
import re

class TDCDTIWrapper():
    def __init__(self,
                TDCDTI):
        self.tdc = TDCDTI
        df_interactions = self.tdc.get_data()
        unique_targets = df_interactions["Target"].unique()
        target_map = {t:i for i, t in enumerate(unique_targets)}
        df_interactions = df_interactions.assign(PROTEIN_ID = df_interactions["Target"].map(target_map))
        df_interactions.columns =["DRUG_ID", "SMILES", "Target_ID", "SEQUENCE", "Y", "PROTEIN_ID"]
        self.drug_smiles = df_interactions.loc[:, ["DRUG_ID", "SMILES"]].drop_duplicates().set_index("DRUG_ID")
        self.protein_sequences = df_interactions.loc[:, ["PROTEIN_ID", "SEQUENCE"]].drop_duplicates().set_index("PROTEIN_ID")
        self.data = df_interactions.loc[:, ["DRUG_ID", "PROTEIN_ID", "Y"]]
    def preprocess(self):
        self.data_subset = self.data
        return self.data_subset
    def get_drugs(self):
        return self.drug_smiles
    def get_proteins(self):
        return self.protein_sequences
    def __str__(self):
        return str(self.tdc.name)

class DTIDatasetManager():
    def __init__(self,
                 processing_pipeline,
                 target_processor,
                 drug_featurizer,
                 protein_featurizer,
                 k=25,
                 partition_column = None,
                 exclude_from_test = [],
                 ):
        self.ppl = processing_pipeline
        self.drug_featurizer = drug_featurizer
        self.protein_featurizer = protein_featurizer
        self.processed_data = self.ppl.preprocess()
        self.splitter = Splitter(self.processed_data, k, partition_column, exclude_from_test)
        self.splitter.fit()
        self.target_processor = target_processor
    def get_data(self):
        return self.processed_data
    def get_partition(self, idx):
        train, val, test = self.splitter[idx]
        self.target_processor.fit(train)
        return self.target_processor.transform(train), self.target_processor.transform(test), self.target_processor.transform(val)
    def get_proteins(self):
        path_cache = f"data/processed/{str(self.ppl)}_{str(self.protein_featurizer)}.pt"
        proteins = self.ppl.get_proteins()
        if os.path.exists(path_cache):
            protein_dict = torch.load(path_cache)
        else:
            protein_dict = protein_dict = self.protein_featurizer(proteins.to_numpy().squeeze(),
                                    proteins.index.to_numpy())
            torch.save(protein_dict, path_cache)
        return protein_dict
    def get_drugs(self):
        smiles =  self.ppl.get_drugs()
        path_cache = f"data/processed/{str(self.ppl)}_{str(self.drug_featurizer)}.pt"
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
    def get_tabular_dataset(self, data, protein_dict, drug_dict):
        protein_df = pd.DataFrame(protein_dict).T
        protein_df.columns = self.ppl.cell_lines.columns
        drugs_df = pd.DataFrame(drug_dict).T
        drugs_df.columns = [f"DRUG_{i}" for i in range(drugs_df.shape[1])]
        return pd.concat([drugs_df.loc[data.loc[:, "DRUG_ID"]].reset_index(drop=True),
                          proteins_df.loc[data.loc[:, "PROTEIN_ID"]].reset_index(drop=True),
                          data.loc[:, ["Y"]].reset_index(drop=True)], axis=1)
import pandas as pd
import requests
import io
import numpy as np
import pickle
import os
from PyDRP.Data import PreprocessingPipeline
from PyDRP.src import Splitter
from PyDRP.Data.features.targets import IdentityPipeline
from PyDRP.Data.features.drugs import GraphCreator
import torch

class TransferDrugsDatasetManager():
    def __init__(self,
                 drugs_processing_pipeline,
                 target_processor,
                 drug_featurizer,
                 k=25,
                 exclude_from_test = [],
                 ):
        self.ppl = drugs_processing_pipeline
        self.drug_featurizer = drug_featurizer
        self.processed_data = self.ppl.preprocess()
        self.splitter = Splitter(self.processed_data, k, "DRUG_ID", exclude_from_test)
        self.splitter.fit(None)
        self.target_processor = target_processor
    def get_data(self):
        return self.processed_data
    def get_partition(self, idx):
        train, val, test = self.splitter[idx]
        self.target_processor.fit(train)
        return self.target_processor.transform(train), self.target_processor.transform(test), self.target_processor.transform(val)
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
    def get_tabular_dataset(self, data, drug_dict):
        drugs_df = pd.DataFrame(drug_dict).T
        drugs_df.columns = [f"DRUG_{i}" for i in range(drugs_df.shape[1])]
        return pd.concat([drugs_df.loc[data.loc[:, "DRUG_ID"]].reset_index(drop=True),
                          data.loc[:, ["Y"]].reset_index(drop=True)], axis=1)
    
class ToxRicPreprocessingPipeline(PreprocessingPipeline):
    def __init__(self, root = "./",
                minimum_experiments = 500,
                filter_missing_ids = True):
        """
        
        """
        if not os.path.exists(root + "data"):
            os.mkdir(root + "data")
        if not os.path.exists(root + "data/raw"):
            os.mkdir(root + "data/raw")
        if not os.path.exists(root + "data/processed"):
            os.mkdir(root + "data/processed")
        self.minimum_experiments = minimum_experiments
        self.filter_missing_ids = filter_missing_ids
        if os.path.exists(root + "data/raw/dfs_toxric.pkl"):
            with open(root + "data/raw/dfs_toxric.pkl", "rb") as f:
                dfs = pickle.load(f)
        else:
            dfs = [pd.read_csv(f"https://toxric.bioinforai.tech/jk/DownloadController/DownloadToxicityInfo?toxicityId={i}") for i in range(102)]
            with open(root + "data/raw/dfs_toxric.pkl", "wb") as f:
                pickle.dump(dfs, f)
        self.dfs = dfs
    def preprocess(self):
        ids = []
        toxs = []
        for i, df in enumerate(self.dfs):
            try:
                ids+=[df.loc[:, ["TAID", "Canonical SMILES"]].set_index("TAID")]
            except:
                pass
            try:
                toxs+=[df.loc[:, ["TAID", "Toxicity Value"]].set_index("TAID").rename(columns={"Toxicity Value": i})]
            except:
                pass
        drug_df = pd.concat(ids).drop_duplicates()
        tox_df = pd.concat(toxs, axis=1).reset_index()
        tox_df = tox_df.loc[:, (~tox_df.isna()).sum(axis=0) > 500].set_index("TAID").drop_duplicates()
        tox_df = tox_df.reset_index().loc[:, ["TAID"]].assign(y = tox_df.to_numpy().tolist())
        tox_df.columns = ["DRUG_ID", "Y"]
        self.data_subset = tox_df
        self.drug_smiles = drug_df
        isin_data = self.data_subset.loc[:, "DRUG_ID"].isin(self.drug_smiles.index.to_numpy())
        self.data_subset = self.data_subset.loc[isin_data]
        return self.data_subset
    def get_drugs(self):
        data_drugs = self.data_subset.loc[:, "DRUG_ID"]
        return self.drug_smiles.loc[data_drugs]
    def __str__(self):
        return "ToxRic_" + str(self.minimum_experiments)
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
import rdkit
from rdkit import rdBase
from PyDRP.Data.utils import TorchGraphsTransferDataset
import torch_geometric
from torch import nn
import torch
import urllib

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
        self.splitter.fit()
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
        dfs_tr = {}
        for i in range(len(self.dfs)):
            try:
                if len(self.dfs[i]["Toxicity Value"].unique()) > 2:
                    dfs_tr[i] = [np.log(self.dfs[i]["Toxicity Value"] + 1)]
            except:
                pass
        log_cols = list(dfs_tr.keys())
        for i, df in enumerate(self.dfs):
            try:
                ids+=[df.loc[:, ["TAID", "Canonical SMILES"]].set_index("TAID")]
            except:
                pass
            if i in log_cols:
                try:
                    df.loc[:, "Toxicity Value"] = np.log(df.loc[:, "Toxicity Value"] + 1)
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
        self.drug_smiles = drug_df.reset_index()
        self.drug_smiles.columns = ["DRUG_ID", "SMILES"]
        self.drug_smiles = self.drug_smiles.set_index("DRUG_ID")
        isin_data = self.data_subset.loc[:, "DRUG_ID"].isin(self.drug_smiles.index.to_numpy())
        self.data_subset = self.data_subset.loc[isin_data]
        return self.data_subset
    def get_drugs(self):
        data_drugs = self.data_subset.loc[:, "DRUG_ID"]
        return self.drug_smiles.loc[data_drugs]
    def __str__(self):
        return "ToxRic_" + str(self.minimum_experiments)

class TDCSingleInstanceWrapper(PreprocessingPipeline):
    """
    Wrapper around tdc.single_pred datasets (https://tdcommons.ai/single_pred_tasks/overview/)
    """
    def __init__(self,
                 TDCsingleInstance,
                 filter_missing_ids = True,
                ):
        self.tdc = TDCsingleInstance
        self.filter_missing_ids = filter_missing_ids
        self.data = self.tdc.get_data()
        self.data.columns = ["DRUG_ID", "SMILES", "Y"]
        self.drug_smiles = self.data.loc[:, ["DRUG_ID", "SMILES"]]
        self.drug_smiles = self.drug_smiles.set_index("DRUG_ID")
    def preprocess(self):
        self.data_subset = self.data.loc[:, ["DRUG_ID", "Y"]]
        if self.filter_missing_ids:
            self.drugs = self.drug_smiles.index.to_numpy()
            self.data_subset = self.data_subset.query("DRUG_ID in @self.drugs")
        return self.data_subset
    def get_drugs(self):
        data_drugs = self.data_subset.loc[:, "DRUG_ID"].unique()
        return self.drug_smiles.loc[data_drugs]
    def __str__(self):
        name = str(self.tdc.name)
        try:
            label_name = str(self.tdc.label_name)
        except:
            label_name = ""
        return f"{name}_{label_name}"

class MultiTaskPreprocessingPipeline(PreprocessingPipeline):
    def __init__(self, preprocessing_pipelines):
        """
        Creates Multitask datasets from a series of preprocessing pipelines.
        preprocessing_pipelines: A list of preprocessing pipelines
        
        """
        self.preprocessing_pipelines = preprocessing_pipelines
    def get_cannonical_smiles(self, smiles, error_on_bad_smiles = False):
        if error_on_bad_smiles:
            return rdkit.Chem.CanonSmiles(smiles)
        try:
            return rdkit.Chem.CanonSmiles(smiles)
        except:
            return smiles
    def preprocess(self):
        dfs = []
        blocker = rdBase.BlockLogs()
        mult_columns = 0
        for i, ppl in enumerate(self.preprocessing_pipelines):
            data_task = ppl.preprocess()
            if type(data_task["Y"].iloc[0]) == list:
                new_cols = pd.DataFrame(data_task['Y'].to_list())
                new_cols.columns = [f"Y_{i+j+mult_columns}" for j in range(new_cols.shape[1])]
                mult_columns += (new_cols.shape[1]-1)
                data_task = pd.concat([data_task.drop("Y", axis=1), new_cols], axis=1)
            else:  
                data_task = data_task.rename(columns = {"Y":f"Y_{i+mult_columns}"})
            drugs_task = ppl.get_drugs().reset_index()
            drugs_task["SMILES"] = drugs_task["SMILES"].apply(self.get_cannonical_smiles)
            task_df = data_task.merge(drugs_task, on = "DRUG_ID").set_index("SMILES")
            task_df = task_df.groupby("SMILES").first()
            task_df["DRUG_ID"] = task_df["DRUG_ID"].astype(str) + f"_{i}"
            dfs += [task_df]
        merged_df = pd.concat(dfs, axis=1)
        merged_df = merged_df.reset_index()
        d_np = merged_df.loc[:, [f"Y_{i}" for i in range(len(self.preprocessing_pipelines) + mult_columns)]].to_numpy()
        merged_df = merged_df.assign(Y = d_np.tolist()).loc[:, ["SMILES", "Y"]].assign(DRUG_ID = np.arange(d_np.shape[0]))
        merged_df = merged_df.loc[:, ["DRUG_ID", "SMILES", "Y"]]
        self.drug_smiles = merged_df.loc[:, ["DRUG_ID", "SMILES"]].set_index("DRUG_ID")
        self.data_subset = merged_df.loc[:, ["DRUG_ID", "Y"]]
        return self.data_subset
    def get_drugs(self):
        blocker = rdBase.BlockLogs()
        data_drugs = self.data_subset.loc[:, "DRUG_ID"]
        return self.drug_smiles.loc[data_drugs]
    def __str__(self):
        strs = [str(ppl) for ppl in self.preprocessing_pipelines]
        return "&".join(strs)

class MakeDrugwise(PreprocessingPipeline):
    def __init__(self, ppl):
        """
        Takes a pairwise ppl and converts it to multitask prediction ppl
        """
        self.ppl = ppl
        self.ppl.preprocess()
    def preprocess(self):
        df = self.ppl.preprocess()
        dwise_df = df.set_index(["DRUG_ID", "CELL_ID"]).unstack()
        Y = dwise_df.to_numpy()
        self.data_subset = dwise_df.drop(dwise_df.columns, axis=1).assign(Y = Y.tolist())
        self.data_subset = self.data_subset.droplevel(0, axis=1)
        self.data_subset.columns = ["Y"]
        self.data_subset =  self.data_subset.reset_index()
        return self.data_subset
    def get_drugs(self):
        self.drug_smiles = self.ppl.get_drugs()
        return self.drug_smiles
    def __str__(self):
        return str(self.ppl) + "_drugwise"

def get_sequential_multitask(train, test, val, drug_dict, threshold = 512, batch_size = 128):
    output = []
    data_matrix_train = np.vstack(train["Y"].apply(np.array).tolist())
    data_matrix_test = np.vstack(test["Y"].apply(np.array).tolist())
    data_matrix_val = np.vstack(val["Y"].apply(np.array).tolist())
    nan_tasks = (~np.isnan(data_matrix_train)).sum(axis=0)
    training_data = nan_tasks >= threshold
    n_tasks = training_data.sum()
    for task in range(n_tasks):
        subtask_df_train = train.assign(Y = data_matrix_train[:, training_data][:, task]).dropna()
        subtask_df_test = test.assign(Y = data_matrix_test[:, training_data][:, task]).dropna()
        subtask_df_val = val.assign(Y = data_matrix_val[:, training_data][:, task]).dropna()
        task_ds_train = TorchGraphsTransferDataset(subtask_df_train, drug_dict)
        task_ds_test = TorchGraphsTransferDataset(subtask_df_test, drug_dict)
        task_ds_val = TorchGraphsTransferDataset(subtask_df_val, drug_dict)
        task_dataloader_train = torch.utils.data.DataLoader(task_ds_train, batch_size = batch_size,
                                                   collate_fn = torch_geometric.data.Batch.from_data_list,
                                                            shuffle=True, drop_last = True)
        task_dataloader_test = torch.utils.data.DataLoader(task_ds_test, batch_size = batch_size,
                                                   collate_fn = torch_geometric.data.Batch.from_data_list)
        task_dataloader_val = torch.utils.data.DataLoader(task_ds_val, batch_size = batch_size,
                                                   collate_fn = torch_geometric.data.Batch.from_data_list)
        is_binary = len(subtask_df_train ["Y"].unique()) < 3
        
        if is_binary:
            w = torch.Tensor([subtask_df_train["Y"].shape[0]/(2 *(subtask_df_train ["Y"] == 1).sum())])
            loss = nn.BCEWithLogitsLoss(pos_weight = w)
        else:
            loss = nn.MSELoss()
        output += [{"train": task_dataloader_train,
                   "test": task_dataloader_test,
                   "val": task_dataloader_val,
                   "loss": loss}]
    return output

class MolDataPreprocessingPipeline():
    def __init__(self, root = "./",
                 threshold = 2000,
                 threshold_imbalance = 0.05,
                 filter_terms = None,
                 filter_missing_ids = True):
        """
        
        """
        if not os.path.exists(root + "data"):
            os.mkdir(root + "data")
        if not os.path.exists(root + "data/raw"):
            os.mkdir(root + "data/raw")
        if not os.path.exists(root + "data/processed"):
            os.mkdir(root + "data/processed")
        self.threshold = threshold
        self.threshold_imbalance = threshold_imbalance
        self.filter_terms = filter_terms
        self.filter_missing_ids = filter_missing_ids
        if not os.path.exists(root + "data/raw/moldata.zip"):
            urllib.request.urlretrieve("https://github.com/LumosBio/MolData/raw/main/Data/all_molecular_data.zip", "data/raw/moldata.zip")
        moldata = pd.read_csv(root + "data/raw/moldata.zip")
        self.drug_smiles = moldata.loc[:, ["smiles", "PUBCHEM_CID"]].drop_duplicates()
        self.drug_smiles.columns = ["SMILES", "DRUG_ID"]
        self.drug_smiles = self.drug_smiles.set_index("DRUG_ID")
        self.data = moldata.set_index("PUBCHEM_CID").iloc[:, 1:]
        aid_disease_mapping = pd.read_csv("https://raw.githubusercontent.com/LumosBio/MolData/main/Data/aid_disease_mapping.csv").set_index("AID")
        self.terms = aid_disease_mapping.columns.to_numpy()
        self.aid_mapping = aid_disease_mapping
        if filter_terms is not None:
            self.columns_term = aid_disease_mapping.query(f"{filter_terms} == 1").index.to_numpy()
        else:
            self.columns_term = None
    def preprocess(self):
        moldata_data = self.data
        if self.columns_term is not None:
            moldata_data = moldata_data.loc[:, self.columns_term]
        num_nonna = (~(moldata_data.isna())).sum(axis=0)
        num_neg = (moldata_data == 0).sum(axis=0)
        passed_filters = ((num_neg/num_nonna) > self.threshold_imbalance) & ((num_neg/num_nonna) < (1-self.threshold_imbalance)) & (num_nonna > self.threshold)
        self.data_subset = self.data[~self.data.loc[:, passed_filters].isna().all(axis=1)].loc[:, passed_filters]
        self.data_subset = self.data_subset.assign(Y = self.data_subset.to_numpy().tolist()).loc[:, "Y"].reset_index()
        self.data_subset.columns = ["DRUG_ID", "Y"]
        return self.data_subset
    def get_drugs(self):
        data_drugs = self.data_subset.loc[:, "DRUG_ID"]
        return self.drug_smiles.loc[data_drugs]
    def __str__(self):
        return f"MolData_{str(self.threshold)}_{str(self.threshold_imbalance)}_{str(self.filter_terms)}"
    
class QmugsPreprocessingPipeline():
    def __init__(self, root = "./"):
        """
        Preprocess the qmugs data giving the computed features.
        """
        if not os.path.exists(root + "data"):
            os.mkdir(root + "data")
        if not os.path.exists(root + "data/raw"):
            os.mkdir(root + "data/raw")
        if not os.path.exists(root + "data/processed"):
            os.mkdir(root + "data/processed")
        self.data_url = "https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2F&files=summary.csv"
        self.data_path = "data/raw/qmugs_raw.csv"
        if not os.path.exists(root + self.data_path):
            urllib.request.urlretrieve(self.data_url, self.data_path)
        self.drug_smiles = pl.scan_csv(data_path).select([pl.col("chembl_id"), pl.col("smiles")]).collect().to_pandas().drop_duplicates()
        self.drug_smiles.columns = ["DRUG_ID", "SMILES"]
        self.drug_smiles = self.drug_smiles.set_index("DRUG_ID")
        self.data = pl.read_csv(data_path).to_pandas().groupby("chembl_id").mean().iloc[:, 3:]
    def preprocess(self):
        data = self.data.copy()
        data = data.loc[:, data.dtypes != bool]
        data = data.assign(Y = data.to_numpy().tolist()).loc[:, "Y"]
        data = data.reset_index()
        data.columns = ["DRUG_ID", "Y"]
        self.data_subset = data
    def get_drugs(self):
        data_drugs = self.data_subset.loc[:, "DRUG_ID"]
        return self.drug_smiles.loc[data_drugs]
    def __str__(self):
        return f"Qmugs"
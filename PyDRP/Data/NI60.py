import requests
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
from PyDRP.src import PreprocessingPipeline
import os
import numpy as np
from io import StringIO
import unlzw3

class NI60PreprocessingPipeline(PreprocessingPipeline):
    def __init__(self, root = "./",
                 target = "GI50",
                 cell_lines = "expression",
                 gene_subset = None,
                filter_missing_ids = True):
        """
        Downloads and preprocesses the data.
        target: Either GI50, TGI, LC50 or IC50
        cell_lines: Data to represent the cell lines. Only expression is implemented.
        gene_subset: A numpy array containing the name of the genes to represent the cell-lines. If None, use all of them.
        """
        if not os.path.exists(root + "data"):
            os.mkdir(root + "data")
        if not os.path.exists(root + "data/raw"):
            os.mkdir(root + "data/raw")
        if not os.path.exists(root + "data/processed"):
            os.mkdir(root + "data/processed")
        self.target = target
        self.root = root
        if target == "GI50":
            if os.path.exists(root + "data/raw/NIGI50.csv"):
                self.data = pd.read_csv(root + "data/raw/NIGI50.csv", index_col = 0)
            else:
                r = requests.get("https://wiki.nci.nih.gov/download/attachments/147193864/GI50.zip?version=5&modificationDate=1672801124000&api=v2")
                z = ZipFile(BytesIO(r.content))
                self.data = pd.read_csv(z.open("GI50.csv"))
                self.data.to_csv(root + "data/raw/NIGI50.csv")
        elif target == "TGI":
            if os.path.exists(root + "data/raw/NITGI.csv"):
                self.data = pd.read_csv(root + "data/raw/NITGI.csv", index_col = 0)
            else:
                r = requests.get("https://wiki.nci.nih.gov/download/attachments/147193864/TGI.zip?version=7&modificationDate=1672801161000&api=v2")
                z = ZipFile(BytesIO(r.content))
                self.data = pd.read_csv(z.open("TGI.csv"))
                self.data.to_csv(root + "data/raw/NITGI.csv")
        elif target == "LC50":
            if os.path.exists(root + "data/raw/NILC50.csv"):
                self.data = pd.read_csv(root + "data/raw/NILC50.csv", index_col = 0)
            else:
                r = requests.get("https://wiki.nci.nih.gov/download/attachments/147193864/LC50.zip?version=5&modificationDate=1672801143000&api=v2")
                z = ZipFile(BytesIO(r.content))
                self.data = pd.read_csv(z.open("LC50.csv"))
                self.data.to_csv(root + "data/raw/NILC50.csv")
        elif target == "IC50":
            if os.path.exists(root + "data/raw/NIIC50.csv"):
                self.data = pd.read_csv(root + "data/raw/NIIC50.csv", index_col = 0)
            else:
                r = requests.get("https://wiki.nci.nih.gov/download/attachments/147193864/IC50.zip?version=5&modificationDate=1672801134000&api=v2")
                z = ZipFile(BytesIO(r.content))
                self.data = pd.read_csv(z.open("IC50.csv"))
                self.data.to_csv(root + "data/raw/NIIC50.csv")
        else:
            raise NotImplementedError
        self.gene_subset = gene_subset
        self.filter_missing_ids = filter_missing_ids
        self.dataset = "NI60"
        if cell_lines == "expression":
            self.get_ccle_expression()
        
        if not os.path.exists(root + "data/processed/smiles_ni.csv"):
            r = requests.get("https://wiki.nci.nih.gov/download/attachments/155844992/NCIOPENC_SMI.BIN?version=1&modificationDate=1378210574000&api=v2")
            z = unlzw3.unlzw(r.content)
            smiles = pd.read_csv(StringIO(z.decode()), sep = "    ", header = None)
            smiles.columns = ["DRUG_ID", "SMILES"]
            smiles.loc[:, "SMILES"] = smiles["SMILES"].str.extract("[0-9]+-[0-9]+-[0-9]+ (.*)")
            self.drug_smiles = smiles.set_index("DRUG_ID")
            self.drug_smiles.to_csv(root + "data/processed/smiles_ni.csv")
        else:
            self.drug_smiles = pd.read_csv(root + "data/processed/smiles_ni.csv")
            # Get rid of incorrect lines
            self.drug_smiles = (self.drug_smiles[self.drug_smiles.loc[:, "DRUG_ID"].apply(len) < 9]
                                .astype({"DRUG_ID":int})
                                .set_index("DRUG_ID"))
    def preprocess(self):
        self.data_subset = self.data.loc[:, ["CELL_NAME", "NSC", "AVERAGE"]]
        self.data_subset = self.data.groupby(["CELL_NAME", "NSC"]).mean().reset_index()
        synonyms = pd.read_csv("https://ndownloader.figshare.com/files/26261569")
        self.data_subset = self.data_subset.merge(synonyms, left_on="CELL_NAME", right_on="cell_line_name").loc[:, ["DepMap_ID", "NSC", "AVERAGE"]]
        self.data_subset.columns = ["CELL_ID", "DRUG_ID", "Y"]
        if self.filter_missing_ids:
            self.lines = self.cell_lines.index.to_numpy()
            self.drugs = self.drug_smiles.index.to_numpy()
            self.data_subset = self.data_subset.query("CELL_ID in @self.lines & DRUG_ID in @self.drugs")
        return self.data_subset.dropna().drop_duplicates()
    def get_cell_lines(self):
        data_lines = self.data_subset.loc[:, "CELL_ID"].unique()
        if self.gene_subset is None:
            return self.cell_lines.loc[data_lines]
        else:
            self.cell_lines = self.cell_lines.loc[:, self.cell_lines.columns.isin(self.gene_subset)]
            return self.cell_lines.loc[data_lines]
    def get_drugs(self):
        data_drugs = self.data_subset.loc[:, "DRUG_ID"].unique()
        return self.drug_smiles.loc[data_drugs]
    
    def __str__(self):
        return "NI60" + self.target
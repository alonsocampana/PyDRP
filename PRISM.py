import pandas as pd
from src import PreprocessingPipeline
import os
import numpy as np

class PRISMPreprocessingPipeline(PreprocessingPipeline):
    def __init__(self, root = "./",
                 target = "ic_50",
                 cell_lines = "expression",
                 gene_subset = None,
                filter_missing_ids = True):
        """
        Downloads and preprocesses the data.
        target: Either ic50, ec50 or AUC
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
        self.gene_subset = gene_subset
        self.filter_missing_ids = filter_missing_ids
        self.dataset = "PRISM"
        if not os.path.exists(root + "data/raw/prism.csv"):
            self.data = pd.read_excel("https://ndownloader.figshare.com/files/20237739")
            self.data.to_csv(root + "data/raw/prism.csv")
        else:
            self.data = pd.read_csv(root + "data/raw/prism.csv", index_col = 0)
        if cell_lines == "expression":
            if os.path.exists(root + "data/processed/prism_expression.csv"):
                self.cell_lines = pd.read_csv(root + "data/processed/prism_expression.csv", index_col=0)
            else:
                self.cell_lines = pd.read_csv("https://ndownloader.figshare.com/files/26261476", index_col=0)
                self.cell_lines.to_csv("data/processed/prism_expression.csv")
            self.cell_lines.columns = self.cell_lines.columns.str.extract("(^[a-zA-Z0-9-\.]+) \(")
        self.drug_smiles = self.data.loc[:, ["name", "smiles"]].drop_duplicates()
        self.drug_smiles.columns = ["DRUG_ID", "SMILES"]
        self.drug_smiles = self.drug_smiles.set_index("DRUG_ID")
    def preprocess(self):
        self.data_subset = self.data.loc[:, ["depmap_id", "name", self.target]].groupby(["depmap_id", "name"]).mean().reset_index()
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
            return self.cell_lines.loc[data_lines, self.gene_subset]
    def get_drugs(self):
        data_drugs = self.data_subset.loc[:, "DRUG_ID"].unique()
        return self.drug_smiles.loc[data_drugs]
    
    def __str__(self):
        return "PRISM"
import requests
from zipfile import ZipFile
from io import BytesIO
import pandas as pd
from PyDRP.src import PreprocessingPipeline
import os
import numpy as np

class CTRPv2PreprocessingPipeline(PreprocessingPipeline):
    def __init__(self, root = "./",
                 target = "ec_50",
                 cell_lines = "expression",
                 gene_subset = None,
                filter_missing_ids = True):
        """
        Downloads and preprocesses the data.
        target: Either ec50 or auc
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
        if target == "ec50":
            self.trgt_col = "apparent_ec50_umol"
        elif target == "auc":
            self.trgt_col = "area_under_curve"
        self.gene_subset = gene_subset
        self.filter_missing_ids = filter_missing_ids
        self.dataset = "CTRPv2"
        if os.path.exists(root + "data/raw/CTRPv2.bytes"):
            f = open(root + "data/raw/CTRPv2.bytes", "rb")
            self.z = ZipFile(f)
        else:
            response = requests.get("https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/CTRPv2.0_2015_ctd2_ExpandedDataset.zip")
            output = open(root + "data/raw/CTRPv2.bytes", "wb")
            output.write(response.content)
            output.close()
            self.z = ZipFile(BytesIO(response.content))
        self.data = pd.read_csv(self.z.open("v20.data.curves_post_qc.txt"), sep = "\t")
        if cell_lines == "expression":
            self.get_ccle_expression()
        df2 = pd.read_csv(self.z.open("v20.meta.per_compound.txt"), sep = "\t")
        self.drug_smiles = df2.loc[:, ["master_cpd_id", "cpd_smiles"]]
        self.drug_smiles.columns = ["DRUG_ID", "SMILES"]
        self.drug_smiles = self.drug_smiles.set_index("DRUG_ID")
    def preprocess(self):
        self.data_subset = self.data.loc[:, ["experiment_id", "master_cpd_id", self.trgt_col]]
        df3 = pd.read_csv(self.z.open("v20.meta.per_cell_line.txt"), sep = "\t")
        synonyms = pd.read_csv("https://ndownloader.figshare.com/files/26261569")
        synonym_table = synonyms.merge(df3, right_on="ccl_name", left_on="stripped_cell_line_name").loc[:, ["DepMap_ID", "master_ccl_id"]].drop_duplicates()
        self.data_subset = self.data_subset.merge(synonym_table, left_on="experiment_id", right_on="master_ccl_id").loc[:, ["DepMap_ID", "master_cpd_id", self.trgt_col]]
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
        return "CTRPv2" + self.target
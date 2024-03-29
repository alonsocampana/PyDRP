import pandas as pd
from PyDRP.src import PreprocessingPipeline
import os
import numpy as np
import requests
from io import BytesIO
import zipfile

class GDSCPreprocessingPipeline(PreprocessingPipeline):
    def __init__(self, root = "./",
                 dataset = "GDSC1",
                 target = "LN_IC50",
                 cell_lines = "expression",
                 gene_subset = None,
                filter_missing_ids = True):
        """
        Downloads and preprocesses the data.
        Dataset: Either GDSC1 or GDSC2
        target: Either LN_IC50 or AUC
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
        self.dataset = dataset
        if dataset == "GDSC1":
            if not os.path.exists(root + "data/raw/gdsc1.csv"):
                self.data = pd.read_excel("https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/GDSC1_fitted_dose_response_24Jul22.xlsx")
                self.data.to_csv(root + "data/raw/gdsc1.csv")
            else:
                self.data = pd.read_csv(root + "data/raw/gdsc1.csv", index_col = 0)
        elif dataset == "GDSC2":
            if not os.path.exists(root + "data/raw/gdsc2.csv"):
                self.data = pd.read_excel("https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/GDSC2_fitted_dose_response_24Jul22.xlsx")
                self.data.to_csv(root + "data/raw/gdsc2.csv")
            else:
                self.data = pd.read_csv(root + "data/raw/gdsc2.csv", index_col = 0)
        df_features = []
        if "expression" in cell_lines:
            if os.path.exists(root + "data/processed/gdsc_expression.csv"):
                expression = pd.read_csv(root + "data/processed/gdsc_expression.csv", index_col = 0).reset_index().astype({"index":int}).set_index("index")
            else:
                data = pd.read_csv("https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip", compression = "zip", sep = "\t")
                data = data.set_index("GENE_SYMBOLS").iloc[:, 1:].T
                data.index = data.index.str.extract("DATA.([0-9]+)").to_numpy().squeeze()
                expression = data.reset_index(drop=False).groupby("index").first()
                expression.to_csv(root + "data/processed/gdsc_expression.csv")
            if self.gene_subset is not None:
                expression = expression.loc[:, expression.columns.isin(self.gene_subset)]
            df_features += [expression]
        if "mutations" in cell_lines:
            if not os.path.exists(root + "data/processed/gdsc_mutations.csv"):
                r = requests.get("https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/BEMs/CellLines/CellLines_CG_BEMs.zip")
                zf = zipfile.ZipFile(BytesIO(r.content))
                mutations = pd.read_csv(zf.open("CellLines_CG_BEMs/PANCAN_SEQ_BEM.txt"), sep = "\t", index_col = 0).T.reset_index().astype({"index":int}).set_index("index")
                mutations.to_csv(root + "data/processed/gdsc_mutations.csv")
            else:
                mutations = pd.read_csv(root + "data/processed/gdsc_mutations.csv", index_col=0).reset_index().astype({"index":int}).set_index("index")
            df_features += [mutations]
        if "cnv" in cell_lines:
            if not os.path.exists(root + "data/processed/gdsc_cnvs.csv"):
                r = requests.get("https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/BEMs/CellLines/CellLines_CNV_BEMs.zip")
                zf = zipfile.ZipFile(BytesIO(r.content))
                cnv = pd.read_csv(zf.open("CellLine_CNV_BEMs/PANCAN_CNA_BEM.rdata.txt"), sep = "\t", index_col = 0).reset_index().astype({"index":int}).set_index("index")
                cnv.to_csv(root + "data/processed/gdsc_cnvs.csv")
            else:
                cnv = pd.read_csv(root + "data/processed/gdsc_cnvs.csv", index_col=0).reset_index().astype({"index":int}).set_index("index")
            df_features += [cnv]
        if "methylation" in cell_lines:
            if not os.path.exists(root + "data/processed/gdsc_methylation.csv"):
                r = requests.get("https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/BEMs/CellLines/CellLines_METH_BEMs.zip")
                zf = zipfile.ZipFile(BytesIO(r.content))
                methylation = pd.read_csv(zf.open("METH_CELLLINES_BEMs/PANCAN.txt"), sep = "\t", index_col = 0).T.reset_index().astype({"index":int}).set_index("index")
                methylation.to_csv(root + "data/processed/gdsc_methylation.csv")
            else:
                methylation = pd.read_csv(root + "data/processed/gdsc_methylation.csv", index_col=0).reset_index().astype({"index":int}).set_index("index")
            df_features += [methylation]
        if "CFE" in cell_lines:
            if not os.path.exists(root + "data/processed/gdsc_fce.csv"):
                r = requests.get("https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/BEMs/CellLines/CellLines_Mo_BEMs.zip")
                zf = zipfile.ZipFile(BytesIO(r.content))
                fce = pd.read_csv(zf.open("CellLines_Mo_BEMs/PANCAN_simple_MOBEM.rdata.tsv"), sep = "\t", index_col = 0).T.reset_index().astype({"index":int}).set_index("index")
                fce.to_csv(root + "data/processed/gdsc_fce.csv")
            else:
                fce = pd.read_csv(root + "data/processed/gdsc_fce.csv").reset_index().astype({"index":int}).set_index("index")
            df_features += [fce]
        self.df_features = df_features
        self.cell_lines = pd.concat(df_features, axis=0).reset_index().groupby("index").max().dropna()
        assert not self.cell_lines.empty, "The resulting cell line features contain no cell-lines. Maybe no valid features were selected"
        self.drug_smiles = pd.read_csv(root + "data/processed/GDSC_smiles.csv", index_col=0)
    def preprocess(self):
        self.data_subset = self.data.loc[:, ["COSMIC_ID", "DRUG_NAME", self.target]].groupby(["COSMIC_ID", "DRUG_NAME"]).mean().reset_index()
        self.data_subset.columns = ["CELL_ID", "DRUG_ID", "Y"]
        if self.filter_missing_ids:
            self.lines = self.cell_lines.index.to_numpy()
            self.drugs = self.drug_smiles.index.to_numpy()
            self.data_subset = self.data_subset.query("CELL_ID in @self.lines & DRUG_ID in @self.drugs")
        return self.data_subset
    
    def get_cell_lines(self):
        data_lines = self.data_subset.loc[:, "CELL_ID"].unique()
        return self.cell_lines.loc[data_lines]
    def get_drugs(self):
        data_drugs = self.data_subset.loc[:, "DRUG_ID"].unique()
        return self.drug_smiles.loc[data_drugs]
    
    def __str__(self):
        return self.dataset

class GDSCRawPreprocessingPipeline(GDSCPreprocessingPipeline):
    def __init__(self, root = "./",
                 dataset = "GDSC1",
                 cell_lines = "expression",
                 gene_subset = None,
                filter_missing_ids = True):
        """
        Downloads and preprocesses the data.
        Dataset: Either GDSC1 or GDSC2
        target: Either LN_IC50 or AUC
        cell_lines: Data to represent the cell lines. Only expression is implemented.
        gene_subset: A numpy array containing the name of the genes to represent the cell-lines. If None, use all of them.
        """
        if not os.path.exists(root + "data"):
            os.mkdir(root + "data")
        if not os.path.exists(root + "data/raw"):
            os.mkdir(root + "data/raw")
        if not os.path.exists(root + "data/processed"):
            os.mkdir(root + "data/processed")
        self.gene_subset = gene_subset
        self.filter_missing_ids = filter_missing_ids
        self.dataset = dataset
        if dataset == "GDSC1":
            if not os.path.exists(root + "data/raw/gdsc1raw.csv"):
                self.data = pd.read_csv("https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/GDSC1_public_raw_data_24Jul22.csv.zip")
                self.data.to_csv(root + "data/raw/gdsc1raw.csv")
            else:
                self.data = pd.read_csv(root + "data/raw/gdsc1raw.csv", index_col = 0)
        elif dataset == "GDSC2":
            if not os.path.exists(root + "data/raw/gdsc2raw.csv"):
                self.data = pd.read_csv("https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/GDSC2_public_raw_data_24Jul22.csv.zip")
                self.data.to_csv(root + "data/raw/gdsc2raw.csv")
            else:
                self.data = pd.read_csv(root + "data/raw/gdsc2raw.csv", index_col = 0)
        if cell_lines == "expression":
            if os.path.exists(root + "data/processed/gdsc_expression.csv"):
                self.cell_lines = pd.read_csv(root + "data/processed/gdsc_expression.csv", index_col = 0)
            else:
                data = pd.read_csv("https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip", compression = "zip", sep = "\t")
                data = data.set_index("GENE_SYMBOLS").iloc[:, 1:].T
                data.index = data.index.str.extract("DATA.([0-9]+)").to_numpy().squeeze()
                self.cell_lines = data.reset_index(drop=False).groupby("index").first()
                self.cell_lines.to_csv(root + "data/processed/gdsc_expression.csv")
        self.drug_smiles = pd.read_csv(root +  "data/processed/GDSC_smiles.csv", index_col=0)
    def preprocess(self):
        self.data_subset = self.data.copy()
        self.data_subset["INTENSITY"] = np.log(self.data_subset["INTENSITY"] + 1)
        identifier_col = self.data_subset.loc[:, "COSMIC_ID"].astype(str) + "&"+ self.data_subset.loc[:, "SCAN_ID"].astype(str) +   self.data_subset.loc[:, "DRUGSET_ID"].astype(str) +  self.data_subset.loc[:, "BARCODE"].astype(str) + self.data_subset.loc[:, "SEEDING_DENSITY"].astype(str)
        self.data_subset = self.data_subset.assign(identifier = identifier_col)
        data_viab = self.data_subset.groupby(["CONC", "identifier", "DRUG_ID", "COSMIC_ID"])["INTENSITY"].median()
        blank_vals = self.data_subset.query("TAG == 'NC-0'").groupby("identifier")["INTENSITY"].median()
        posblank_vals = self.data_subset.query("TAG == 'B'").groupby("identifier")["INTENSITY"].median()
        data_viab = data_viab.reset_index()
        max_vals = blank_vals.loc[data_viab.loc[:, "identifier"]]
        min_vals = posblank_vals.loc[data_viab.loc[:, "identifier"]]
        data_viab["INTENSITY"] = (data_viab["INTENSITY"].to_numpy() - min_vals.to_numpy())/(max_vals.to_numpy().squeeze() - min_vals.to_numpy().squeeze())
        data_viab = data_viab.groupby(["DRUG_ID", "COSMIC_ID", "CONC"])["INTENSITY"].mean().reset_index()
        matrix_viab = data_viab.set_index(["DRUG_ID", "COSMIC_ID", "CONC"])["INTENSITY"].reset_index().pivot(index=["DRUG_ID", "COSMIC_ID"], columns = "CONC", values= "INTENSITY")
        Y = matrix_viab.to_numpy().tolist()
        self.data_subset = matrix_viab.reset_index().loc[:, ["DRUG_ID", "COSMIC_ID"]].assign(Y = Y)
        self.data_subset.columns = ["DRUG_ID", "CELL_ID", "Y"]
        self.drugs_table = pd.read_csv("data/processed/gdscidtoname.csv", index_col=0)
        self.data_subset = self.data_subset.merge(self.drugs_table, on="DRUG_ID").loc[:, ["CELL_ID", "DRUG_NAME", "Y"]]
        self.data_subset.columns = ["CELL_ID", "DRUG_ID", "Y"]
        if self.filter_missing_ids:
            self.lines = self.cell_lines.index.to_numpy()
            self.drugs = self.drug_smiles.index.to_numpy()
            self.data_subset = self.data_subset.query("CELL_ID in @self.lines & DRUG_ID in @self.drugs") 
        return self.data_subset
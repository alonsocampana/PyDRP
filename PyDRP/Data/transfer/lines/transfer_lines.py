import pandas as pd
import requests
import io
import numpy as np
import urllib.request 
import os
import polars as pl
from PyDRP.src import Splitter

class TransferLinesDatasetManager():
    def __init__(self,
                 lines_processing_pipeline,
                 target_processor,
                 line_featurizer,
                 k=25,
                 exclude_from_test = [],
                 ):
        self.ppl = lines_processing_pipeline
        self.line_featurizer = line_featurizer
        self.processed_data = self.ppl.preprocess()
        self.splitter = Splitter(self.processed_data, k, "CELL_ID", exclude_from_test)
        self.splitter.fit()
        self.target_processor = target_processor
    def get_data(self):
        return self.processed_data
    def get_partition(self, idx):
        train, val, test = self.splitter[idx]
        self.target_processor.fit(train)
        return self.target_processor.transform(train), self.target_processor.transform(test), self.target_processor.transform(val)
    def get_cell_lines(self):
        cell_lines = self.ppl.get_cell_lines()
        return self.line_featurizer(cell_lines.to_numpy(),
                                    cell_lines.index.to_numpy())
    def get_tabular_dataset(self, data, line_dict):
        lines_df = pd.DataFrame(line_dict).T
        lines_df.columns = self.ppl.cell_lines.columns
        return pd.concat([lines_df.loc[data.loc[:, "CELL_ID"]].reset_index(drop=True),
                          data.loc[:, ["Y"]].reset_index(drop=True)], axis=1)

class GTEXPreprocessingPipeline():
    def __init__(self,
                 dataset = "gtextcgatarget",
                filter_missing_ids = True):
        self.dataset = dataset
        target_path = "data/raw/TcgaTargetGtex_rsem_gene_fpkm.gz"
        if os.path.exists(target_path):
            pass
        else:
            urllib.request.urlretrieve("https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/TcgaTargetGtex_rsem_gene_fpkm.gz",target_path)
        expression = pl.read_csv(target_path, sep="\t")
        gdsc_exp = pd.read_csv("data/processed/gdsc_expression.csv")
        table_equivalence = pd.read_csv("data/gdsc_gtex_genes.csv", index_col=0)
        genes = expression["sample"]
        genes_srs = genes.to_pandas()
        samples = expression.columns[1:]
        exp = expression.drop("sample").transpose().to_numpy().astype(np.float32)
        genes_str_xt = genes_srs.str.extract("([0-9a-zA-Z]+)").squeeze()
        isin_table = genes_str_xt.isin(table_equivalence["Gene stable ID"].to_numpy()).to_numpy().squeeze()
        genes_exp = genes_str_xt[isin_table]
        genes_pcman = table_equivalence.set_index("Gene stable ID").loc[genes_exp].squeeze()
        cell_lines = pd.DataFrame(exp[:, isin_table], columns = genes_pcman, index=samples)
        url_labels = "https://kidsfirstxena.s3.us-east-1.amazonaws.com/download/TCGA_target_GTEX_KF%2Fphenotype.txt"
        path_labels = "data/raw/labels_gdsc_gtex.csv"
        if os.path.exists(path_labels):
            labels = pd.read_csv(path_labels, index_col=0)
        else:
            labels = pd.read_csv(url_labels, encoding = "latin1", error_bad_lines=False, sep = "\t")
            labels.to_csv(path_labels)
        self.data = labels
        self.cell_lines = cell_lines
        cell_line_ids = self.cell_lines.index.to_numpy()
        isin_data = self.data["sample"].isin(cell_line_ids)
        self.data = self.data[isin_data]
    def preprocess(self):
        labels = self.data.copy()
        labels = labels.set_index("sample")
        labels_type = (labels["_sample_type"] == "Normal Tissue").astype(int)
        labels["_sample_type"] = labels_type
        labels_counts = labels["_primary_site"].value_counts()
        infrequent_sites = labels_counts[labels_counts< 50].index.to_numpy()
        is_infrequent = labels["_primary_site"].isin(infrequent_sites)
        labels["_primary_site"][is_infrequent] = "other"
        counts_sites = labels["_primary_site"].value_counts()
        count_srs = counts_sites.reset_index().assign(rank = np.arange(len(counts_sites))).set_index("index")["rank"]
        map_int = count_srs.to_dict()
        labels["_primary_site"] = labels["_primary_site"].map(map_int)
        labels["_gender"] = (labels["_gender"] == "female").astype(int)
        classification_df = pd.concat([pd.get_dummies(labels["_primary_site"]), labels["_gender"], labels["_sample_type"]], axis=1)
        classification_df = classification_df.assign(Y = classification_df.to_numpy().tolist()).loc[:, ["Y"]].reset_index()
        classification_df.columns = ["CELL_ID", "Y"]
        self.data_subset = classification_df
        return self.data_subset
    def get_cell_lines(self):
        data_lines = self.data_subset.loc[:, "CELL_ID"].unique()
        return self.cell_lines.loc[data_lines]
    def __str__(self):
        return str(self.dataset)
import polars as pl
import pandas as pd
from urllib.request import urlretrieve
import gzip
import shutil
import uuid
import os
import numpy as np

class STRINGInterface():
    def __init__(self,
                 name,
                 threshold = 700,
                 organism_id = None,
                 remove_duplicates = True,
                 filter_missing_ids = True):
        self.organism_id = organism_id
        self.name = name
        self.data_path = f"data/raw/string_{name}.txt.gz"
        self.sequences_path = f"data/raw/string_{name}_seqs.csv"
        self.data_url = data_url = f"https://stringdb-static.org/download/protein.links.full.v11.5/{self.organism_id}.protein.links.full.v11.5.txt.gz"
        self.sequences_url = f"https://stringdb-static.org/download/protein.sequences.v11.5/{self.organism_id}.protein.sequences.v11.5.fa.gz"
        self.threshold = threshold
        self.remove_duplicates = remove_duplicates
        self.filter_missing_ids = filter_missing_ids
    def preprocess(self):
        if not os.path.exists(self.data_path):
            urlretrieve(self.data_url, self.data_path)
        else:
            pass
        file = gzip.GzipFile(self.data_path, "r")
        with gzip.open(self.data_path, 'rb') as f_in:
            temp_file= f"temp_{str(uuid.uuid1())}.txt'"
            full_path = f'data/temp/{temp_file}'
            with open(full_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            try:
                string_df = (pl.scan_csv(full_path, sep = " ")
                             .filter(pl.col('combined_score') >= self.threshold)
                             .select([pl.col('protein1'), pl.col('protein2'), pl.col('combined_score')])
                             .collect()
                             .to_pandas())
                string_df.columns = ["PROTEIN_ID", "PROTEIN_ID", "Y"]
            except:
                pass
            finally:
                os.remove(full_path)
        if not os.path.exists(self.sequences_path):
            seqs = pd.read_csv(self.sequences_url, lineterminator=">", header = None)
            regex = f"({self.organism_id}\.[A-Z0-9]+[0-9\._]+)\n"
            seqs = seqs[0].str.split(regex,expand = True).iloc[:, 1:]
            seqs.columns = ["PROTEIN_ID", "SEQUENCE"]
            seqs.to_csv(self.sequences_path)
        else:
            seqs = pd.read_csv(self.sequences_path)
        self.protein_sequences = seqs.set_index("PROTEIN_ID")
        if self.remove_duplicates:
            df = string_df.iloc[:, :2]
            df1 = pd.DataFrame(np.sort(df.values, axis=1), index=df.index, columns=df.columns)
            string_df = string_df[~df1.duplicated()]
        if self.filter_missing_ids:
            is1in = string_df.iloc[:, 0].isin(seqs["PROTEIN_ID"].to_numpy())
            is2in = string_df.iloc[:, 1].isin(seqs["PROTEIN_ID"].to_numpy())
            string_df = string_df.loc[is1in&is2in]
        self.data_subset = string_df
        return self.data_subset
    def get_proteins(self):
        unique_prots = np.union1d(self.data_subset.iloc[:, 0].unique(), self.data_subset.iloc[:, 1].unique())
        return self.protein_sequences.loc[unique_prots]
    def __str__(self):
        return f"STRING{self.organism_id}" + str(self.threshold)
class STRINGHuman(STRINGInterface):
    def __init__(self, threshold = 700):
        self.organism_id
        super().__init__(name = "human",
                 organism_id = 9606,
                 threshold = threshold,)
class STRINGMice(STRINGInterface):
    def __init__(self, threshold = 700):
        super().__init__(name = "mice",
                 threshold = threshold,
                 organism_id = 10090)
class STRINGArabidopsis(STRINGInterface):
    def __init__(self, threshold = 700):
        super().__init__(name = "arabidopsis",
                         threshold = threshold,
                        organism_id = 3702,)
    
class STRINGPseudomonas(STRINGInterface):
    def __init__(self, threshold = 700):
        super().__init__(name = "pseudomonas",
                         threshold = threshold,
                        organism_id = 208964,)
def get_species_STRING():
    return pd.read_csv("https://stringdb-static.org/download/species.v11.5.txt", sep = "\t")
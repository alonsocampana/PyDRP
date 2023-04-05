from transformers import BertForMaskedLM, BertTokenizer, pipeline
import gc
import torch
import numpy as np
import pandas as pd
import re

class BertProteinFeaturizer():
    def __init__(self,
                 as_array = False,
                 device = torch.device("cuda"),
                 sequence_length = 2048):
        self.device = device
        self.as_array = as_array
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
        self.model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
        self.model.eval()
        self.model.to(device)
        self.sequence_length = sequence_length
    def __call__(self, protein_seqs, protein_ids):
        embeds = {}
        sequences_spaced = (pd.Series(protein_seqs)
                            .apply(lambda x: " ".join(list(x)))
                            .apply(lambda x: re.sub(r"[UZOB]", "X", x))
                            .tolist())
        #tokens = self.tokenizer.batch_encode_plus(sequences_spaced, add_special_tokens=True, pad_to_max_length=True)
        for i in range(len(sequences_spaced)):
            with torch.no_grad():
                tokens = self.tokenizer(sequences_spaced[i], add_special_tokens=True)
                input_ids = torch.Tensor(tokens["input_ids"]).long().to(self.device).unsqueeze(0)
                attention_mask = torch.Tensor(tokens["attention_mask"]).bool().to(self.device).unsqueeze(0)
                embedding = self.model(output_hidden_states = True,
                                  input_ids=input_ids,
                                  attention_mask=attention_mask)["hidden_states"][-1]
                if embedding.shape[1] < self.sequence_length:
                    diff = self.sequence_length - embedding.shape[1]
                    padding = embedding.new_zeros([embedding.shape[0], diff, embedding.shape[2]])
                    embedding = torch.cat([embedding, padding], axis=1)
                elif embedding.shape[1] > self.sequence_length:
                    embedding = embedding[:, :self.sequence_length, :]
                if self.as_array:
                    embedding = np.array(embedding)
                embeds[protein_ids[i]] = embedding.detach().cpu().squeeze()
        return embeds
    def __str__(self):
        return "ProtBert" + f"_{self.sequence_length}"
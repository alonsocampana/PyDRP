from src import DrugFeaturizer
import numpy as np
from rdkit.Chem import AllChem

class ECFPFeaturizer(DrugFeaturizer):
    def __init__(self, R=2, L= 2**10, use_features=False, use_chirality = False, transform = None):
        self.R = R
        self.L = L
        self.use_features = use_features
        self.use_chirality = use_chirality
        self.transform = transform
    def __call__(self, smiles_list, drugs = None):
        drug_dict = {}
        if drugs is None:
            drugs = np.arange(len(smiles_list))
        for i in range(len(smiles_list)):
            smiles = smiles_list[i]
            molecule = AllChem.MolFromSmiles(smiles)
            feature_list = AllChem.GetMorganFingerprintAsBitVect(molecule,
                                                            radius = self.R,
                                                            nBits = self.L,
                                                            useFeatures = self.use_features,
                                                            useChirality = self.use_chirality)
            f = np.array(feature_list)
            if self.transform is not None:
                f = self.transform(f)
            drug_dict[drugs[i]] = f
        return drug_dict
    def __str__(self):
        """
        returns a description of the featurization
        """
        return "ECFinterprint"
        
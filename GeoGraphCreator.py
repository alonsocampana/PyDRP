import pandas as pd
import numpy as np
import base64
import multiprocessing as mp
import re
import warnings
# Pytorch and Pytorch Geometric
from torch_geometric.data import Data, Batch
import torch_geometric
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric import transforms as T
from torch_geometric.utils import coalesce
# RDkit
import rdkit
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG
from rdkit.Chem import rdMolDescriptors

from src import DrugFeaturizer
from io import StringIO
from scipy.spatial import KDTree

def one_hot_encoding(x, permitted_list):
        """
        Maps input elements x which are not in the permitted list to the last element
        of the permitted list.
        """
        if x not in permitted_list:
            x = permitted_list[-1]
        binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
        return binary_encoding

def get_atom_features(atom, 
                          use_chirality = True, 
                          hydrogens_implicit = True):
        """
        Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
        """
        # define list of permitted atoms

        permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown']

        if hydrogens_implicit == False:
            permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

        # compute atom features

        atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)

        n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])

        formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])

        hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])

        is_in_a_ring_enc = [int(atom.IsInRing())]

        is_aromatic_enc = [int(atom.GetIsAromatic())]

        atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]

        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]

        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
        atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled

        if use_chirality == True:
            chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
            atom_feature_vector += chirality_type_enc

        if hydrogens_implicit == True:
            n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
            atom_feature_vector += n_hydrogens_enc
        return np.array(atom_feature_vector)

def get_bond_features(bond, 
                          use_stereochemistry = True):
        """
        Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
        """
        permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)

        bond_is_conj_enc = [int(bond.GetIsConjugated())]

        bond_is_in_ring_enc = [int(bond.IsInRing())]

        bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

        if use_stereochemistry == True:
            stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
            bond_feature_vector += stereo_type_enc
        return np.array(bond_feature_vector)

class GeoGraphCreator(DrugFeaturizer):
    def __init__(self, knn = 4):
        self.knn = knn
    def __call__(self, smiles_list, drugs = None, **kwargs):
        if drugs is None:
            drugs = np.arange(0, len(smiles_list))
        data_dict = {}
        for x, drug in enumerate(drugs):
                try:
                    mol = Chem.MolFromSmiles(smiles_list[x])
                    mol = Chem.rdmolops.AddHs(mol)
                    AllChem.EmbedMultipleConfs(mol,useRandomCoords=True, maxAttempts = 500)
                    Chem.rdmolops.RemoveStereochemistry(mol)
                    AllChem.MMFFOptimizeMolecule(mol)
                    mol = Chem.rdmolops.RemoveHs(mol)
                    df_coords = pd.read_csv(StringIO(Chem.MolToXYZBlock(mol)), header = None, skiprows=1, engine = "python", sep = "[\s]+").set_index(0)
                    kdt = KDTree(df_coords)
                    NN = self.knn
                    nn_qry = kdt.query(df_coords, NN + 1)
                    es = []
                    ds = []
                    ns = []
                    for n in range(1, NN):
                        es += [nn_qry[1][:, [0, n]]]
                        ds += [nn_qry[0][:, n]]
                        ns += [np.ones(len(nn_qry[1])) * n]
                    es = np.vstack(es).T
                    ds = np.concatenate(ds)
                    ns = np.concatenate(ns)
                    FS2 = np.concatenate([ds[:, None], ns[:, None]], axis=1)
                    X = np.zeros([mol.GetNumAtoms(), 79])
                    for atom in mol.GetAtoms():
                        X[atom.GetIdx(), :] = get_atom_features(atom)
                    es_bond = np.nonzero(GetAdjacencyMatrix(mol))
                    (rows, cols) = es_bond
                    EF = np.zeros((2*mol.GetNumBonds(), 10))
                    FS = np.zeros((2*mol.GetNumBonds(), 2))
                    for (k, (i,j)) in enumerate(zip(rows, cols)):
                        EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
                    EF2 = np.zeros((len(ds), 10))
                    es = torch.Tensor(np.hstack([es, es_bond]))
                    FS = torch.cat([torch.Tensor(np.hstack([EF, FS])),
                    torch.Tensor(np.hstack([EF2, FS2]))])
                    es, FS = coalesce(es, FS)
                    data_dict[drug] = Data(x = X, edge_index = es, edge_attr = FS)
                except:
                    warnings.warn( f"{smiles_list[x]} could not be transformed into a graph", RuntimeWarning,)
        return data_dict
    def __str__(self):
        suffix = f"geo_graphs{self.knn}-NN"
        return suffix
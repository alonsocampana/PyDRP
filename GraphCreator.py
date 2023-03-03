import pandas as pd
import numpy as np
import base64
import multiprocessing as mp
import re
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


class BruteLine():
    def __init__(self):
        pass
    def __call__(self, data):
        N = data.num_nodes
        edge_index, edge_attr = data.edge_index, data.edge_attr
        edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)
        (row, col) = edge_index
        ori = []
        e = []
        for i, t in enumerate(col):
            for j, s in enumerate(row):
                if (i != j) & t == s:
                    ori += [t.item()]
                    e += [[i, j]]
        for i, t in enumerate(col):
            for j, s in enumerate(col):
                if (i != j) & t == s:
                    ori += [t.item()]
                    e += [[i, j]]
        for i, t in enumerate(row):
            for j, s in enumerate(row):
                if (i != j) & t == s:
                    ori += [t.item()]
                    e += [[i, j]]
        pointers = torch.Tensor(ori)
        try:
            es, pointers = torch_geometric.utils.remove_self_loops(torch.Tensor(e).T, pointers)
        except:
            es = torch.Tensor(e).T
        es, pointers = coalesce(es, edge_attr = pointers.unsqueeze(1), reduce= "min")
        data["pointers_lg"] = pointers.long()
        data["es_lg"] = es.long()
        return data

class GraphCreator(DrugFeaturizer):
    def __init__(self, use_supernode = False, 
                       add_linegraph = False,):
        self.use_supernode = use_supernode
        self.add_linegraph = add_linegraph
    def one_hot_encoding(self, x, permitted_list):
        """
        Maps input elements x which are not in the permitted list to the last element
        of the permitted list.
        """
        if x not in permitted_list:
            x = permitted_list[-1]
        binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
        self.bl = BruteLine()
        return binary_encoding


    def get_atom_features(self, atom, 
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

        atom_type_enc = self.one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)

        n_heavy_neighbors_enc = self.one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])

        formal_charge_enc = self.one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])

        hybridisation_type_enc = self.one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])

        is_in_a_ring_enc = [int(atom.IsInRing())]

        is_aromatic_enc = [int(atom.GetIsAromatic())]

        atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]

        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]

        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
        atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled

        if use_chirality == True:
            chirality_type_enc = self.one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
            atom_feature_vector += chirality_type_enc

        if hydrogens_implicit == True:
            n_hydrogens_enc = self.one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
            atom_feature_vector += n_hydrogens_enc
        return np.array(atom_feature_vector)

    def get_bond_features(self, bond, 
                          use_stereochemistry = True):
        """
        Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
        """
        permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        bond_type_enc = self.one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)

        bond_is_conj_enc = [int(bond.GetIsConjugated())]

        bond_is_in_ring_enc = [int(bond.IsInRing())]

        bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

        if use_stereochemistry == True:
            stereo_type_enc = self.one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
            bond_feature_vector += stereo_type_enc
        return np.array(bond_feature_vector)

    def __call__(self, smiles_list, drugs = None, **kwargs):
        use_supernode = self.use_supernode
        add_linegraph = self.add_linegraph
        if drugs is None:
            drugs = np.arange(0, len(smiles_list))
        data_dict = {}

        for x, drug in enumerate(drugs):
            try:
                # convert SMILES to RDKit mol object
                smiles = smiles_list[x]
                mol = Chem.MolFromSmiles(smiles)
                # get feature dimensions
                n_nodes = mol.GetNumAtoms()
                n_edges = 2*mol.GetNumBonds()
                unrelated_smiles = "O=O"
                unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
                n_node_features = len(self.get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
                n_edge_features = len(self.get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
                # construct node feature matrix X of shape (n_nodes, n_node_features)
                X = np.zeros((n_nodes, n_node_features))
                for atom in mol.GetAtoms():
                    X[atom.GetIdx(), :] = self.get_atom_features(atom)

                X = torch.tensor(X, dtype = torch.float)

                # construct edge index array E of shape (2, n_edges)
                (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
                torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
                torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
                E = torch.stack([torch_rows, torch_cols], dim = 0)

                # construct edge feature array EF of shape (n_edges, n_edge_features)
                EF = np.zeros((n_edges, n_edge_features))

                for (k, (i,j)) in enumerate(zip(rows, cols)):

                    EF[k] = self.get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))

                EF = torch.tensor(EF, dtype = torch.float)
                add_f = {kwarg: torch.Tensor([kwargs[kwarg][x]]) for kwarg in kwargs.keys()}
                if use_supernode:
                    super_node = torch.zeros([1, X.shape[1]])
                    trgt_supernode = X.shape[0]
                    extra_indices = torch.cat([torch.arange(0, X.shape[0])[:, None], 
                                         torch.full([X.shape[0], 1],  trgt_supernode)], axis=1).T
                    extra_f = torch.zeros([extra_indices.shape[1], EF.shape[1]])
                    indicator_indices = torch.cat([torch.zeros([E.shape[1], 1]),
                                               torch.ones([extra_indices.shape[1], 1])], axis=0)
                    X = torch.cat([X, super_node], axis=0)
                    E = torch.cat([E, extra_indices], axis=1)
                    EF = torch.cat([indicator_indices, torch.cat([EF,
                                                        extra_f], axis=0)], axis=1)
                data_dict[drug] = Data(x = X, edge_index = E, edge_attr = EF, **add_f)
                if add_linegraph:
                    data_dict[drug] = self.bl(data_dict[drug])
            except Exception as e:
                warnings.warn( f"{smiles_list[x]} could not be transformed into a graph", RuntimeWarning,)
        return data_dict
    def __str__(self):
        suffix = "graphs"
        if self.add_linegraph:
            suffix += "_linegraph"
        if self.use_supernode:
            suffix += "_supernode"
        return suffix
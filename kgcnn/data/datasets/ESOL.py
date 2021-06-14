import os

import numpy as np
import pandas as pd

from kgcnn.data.base import GraphDatasetBase
from kgcnn.data.mol.molgraph import smile_to_graph

class ESOL(GraphDatasetBase):

    data_main_dir = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets")
    data_directory = "ESOL"
    download_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
    file_name = 'delaney-processed.csv'
    unpack_tar = False
    unpack_zip = False
    unpack_directory = None
    fits_in_memory = True

    def read_in_memory(self):

        filepath = os.path.join(self.data_main_dir, self.data_directory, self.file_name)
        data = pd.read_csv(filepath)
        self.data =  data
        self.data_keys = data.columns


    def get_graph(self):

        smiles = self.data['smiles'].values

        # Choose node feautres:
        nf = ['AtomicNum', 'NumExplicitHs', 'NumImplicitHs', 'IsAromatic', 'TotalDegree', 'TotalValence', 'Mass',
              'IsInRing', 'Hybridization', 'ChiralTag', 'FormalCharge', 'ImplicitValence', 'NumRadicalElectrons']
        ef = ['BondType', 'IsAromatic', 'IsConjugated', 'IsInRing', 'Stereo']
        sf = ['ExactMolWt', 'NumAtoms']

        graph_state = []
        nodes = []
        edges = []
        edge_indices = []
        for i,sm in enumerate(smiles):
            atom_sym, atom_info, atom_pos, bond_idx, bond_info, mol_info = smile_to_graph(sm, nodes=nf,
                                                                                          edges=ef, state=sf)
            nodes.append(atom_info)
            edges.append(bond_info)
            edge_indices.append(np.array(bond_idx, dtype="int64"))
            graph_state.append(np.array(mol_info, dtype="float32"))

        # Prepare graph state for all molecules as np.array
        graph_state = np.array(graph_state, dtype="float32" )

        # One hot encoding
        one_hot_list = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "Te", "I", "At"]


        return atom_sym, atom_info, atom_pos, bond_idx, bond_info, mol_info







dataset = ESOL()
graph = dataset.get_graph()
import os
import numpy as np
import pandas as pd

from kgcnn.data.base import GraphDatasetBase
from kgcnn.data.mol.molgraph import smile_to_graph


class ESOLDataset(GraphDatasetBase):
    """Store and process full ESOL dataset."""

    data_main_dir = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets")
    data_directory = "ESOL"
    download_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
    file_name = 'delaney-processed.csv'
    unpack_tar = False
    unpack_zip = False
    unpack_directory = None
    fits_in_memory = True

    def __init__(self, reload=False, verbose=1):
        """Initialize ESOL dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        self.data = None
        self.data_keys = None
        # Use default base class init()
        super(ESOLDataset, self).__init__(reload=reload, verbose=verbose)

    def read_in_memory(self, verbose=1):
        """Load ESOL data into memory and already split into items.

        Args:
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        filepath = os.path.join(self.data_main_dir, self.data_directory, self.file_name)
        data = pd.read_csv(filepath)
        self.data = data
        self.data_keys = data.columns

    def get_graph(self, verbose=1):
        """Make graph tensor objects for ESOL from smiles. Requires rdkit installed.

        Args:
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.

        Returns:
            tuple: labels, nodes, edges, edge_indices, graph_state
        """
        labels2 = np.expand_dims(np.array(self.data['measured log solubility in mols per litre']), axis=-1)
        # labels1 = np.expand_dims(np.array(self.data['ESOL predicted log solubility in mols per litre']), axis=-1)
        # labels = np.concatenate([labels1, labels2], axis=-1)
        labels = labels2

        if verbose > 0:
            print("INFO: Making graph...", end='', flush=True)
        smiles = self.data['smiles'].values

        # Choose node feautres:
        nf = ['AtomicNum', 'NumExplicitHs', 'NumImplicitHs', 'IsAromatic', 'TotalDegree', 'TotalValence', 'Mass',
              'IsInRing', 'Hybridization', 'ChiralTag', 'FormalCharge', 'ImplicitValence', 'NumRadicalElectrons']
        ef = ['BondType', 'IsAromatic', 'IsConjugated', 'IsInRing', 'Stereo']
        sf = ['ExactMolWt', 'NumAtoms']

        graph_state = []
        atom_label = []
        nodes = []
        edges = []
        edge_indices = []
        for i, sm in enumerate(smiles):
            atom_sym, atom_pos, atom_info, bond_idx, bond_info, mol_info = smile_to_graph(sm, nodes=nf,
                                                                                          edges=ef, state=sf)
            nodes.append(np.array(atom_info, dtype="float32"))
            edges.append(np.array(bond_info, dtype="float32"))
            edge_indices.append(np.array(bond_idx, dtype="int64"))
            graph_state.append(np.array(mol_info, dtype="float32"))
            atom_label.append(atom_sym)

        # Prepare graph state for all molecules as a single np.array
        graph_state = np.array(graph_state, dtype="float32")

        # One-hot encoding
        one_hot_list = ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S']
        for i, mol in enumerate(atom_label):
            for j, at in enumerate(mol):
                atom_label[i][j] = [1 if x == at else 0 for x in one_hot_list]
        atom_label = [np.array(x, dtype='float32') for x in atom_label]

        # Add one-hot encoding to node features
        for i, x in enumerate(nodes):
            nodes[i] = np.concatenate([atom_label[i], x], axis=-1)

        if verbose > 0:
            print("done")

        return labels, nodes, edges, edge_indices, graph_state

import os
import numpy as np
import pandas as pd

from kgcnn.data.base import GraphDatasetBase
from kgcnn.mol.molgraph import MolecularGraph, OneHotEncoder

import rdkit.Chem as Chem


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

        if verbose > 0:
            print("INFO: Making graph...", end='', flush=True)
        smiles = self.data['smiles'].values

        # Choose node feautres:
        nf = ['Symbol', 'TotalDegree', 'FormalCharge', 'NumRadicalElectrons', 'Hybridization',
              'IsAromatic', 'IsInRing', 'TotalNumHs', 'CIPCode', "ChiralityPossible", "ChiralTag"]
        ef = ['BondType', 'Stereo', 'IsAromatic', 'IsConjugated', 'IsInRing', "Stereo"]
        sf = ['ExactMolWt', 'NumAtoms']
        encoder = {
            "Symbol": OneHotEncoder(['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At']),
            "Hybridization": OneHotEncoder([Chem.rdchem.HybridizationType.SP,
                                            Chem.rdchem.HybridizationType.SP2,
                                            Chem.rdchem.HybridizationType.SP3,
                                            Chem.rdchem.HybridizationType.SP3D,
                                            Chem.rdchem.HybridizationType.SP3D2]),
            "TotalDegree": OneHotEncoder([0, 1, 2, 3, 4, 5], add_others=False),
            "TotalNumHs": OneHotEncoder([0, 1, 2, 3, 4], add_others=False),
            "BondType": OneHotEncoder([Chem.rdchem.BondType.SINGLE,
                                       Chem.rdchem.BondType.DOUBLE,
                                       Chem.rdchem.BondType.TRIPLE,
                                       Chem.rdchem.BondType.AROMATIC], add_others=False),
            "Stereo": OneHotEncoder([Chem.rdchem.BondStereo.STEREONONE,
                                     Chem.rdchem.BondStereo.STEREOANY,
                                     Chem.rdchem.BondStereo.STEREOZ,
                                     Chem.rdchem.BondStereo.STEREOE], add_others=False),
            "CIPCode": OneHotEncoder(['R', 'S'], add_others=False)}

        graph_state = []
        nodes = []
        edges = []
        edge_indices = []
        labels = []

        for i, sm in enumerate(smiles):
            mg = MolecularGraph(sm, add_Hs=False)
            mg.make_features(nodes=nf, edges=ef, state=sf, encoder=encoder)
            atom_info, bond_info, bond_idx, mol_info = mg.atom_features, mg.bond_features, mg.bond_indices, mg.molecule_features

            if len(bond_idx) > 0:
                nodes.append(np.array(atom_info, dtype="float32"))
                edges.append(np.array(bond_info, dtype="float32"))
                edge_indices.append(np.array(bond_idx, dtype="int64"))
                graph_state.append(np.array(mol_info, dtype="float32"))
                labels.append(labels2[i])

        # Prepare graph state for all molecules as a single np.array
        graph_state = np.array(graph_state, dtype="float32")

        labels = np.array(labels)

        if verbose > 0:
            print("done")
            for key, value in encoder.items():
                print("INFO: OneHotEncoder", key, "found", value.found_values)

        return labels, nodes, edges, edge_indices, graph_state

# ed = ESOLDataset()
# labels, nodes, edges, edge_indices, graph_state = ed.get_graph()

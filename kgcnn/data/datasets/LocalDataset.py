import os
import numpy as np
import pandas as pd

from kgcnn.data.moleculenet import MoleculeNetDataset
from kgcnn.mol.molgraph import MolecularGraphRDKit, OneHotEncoder
from kgcnn.utils.data import save_json_file

import rdkit.Chem as Chem


class LocalDataset(MoleculeNetDataset):
    """Store and process full local dataset."""

    data_main_dir = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets")
    unpack_tar = False
    unpack_zip = False
    unpack_directory = None
    fits_in_memory = True
    require_prepare_data = True

    def __init__(self, dataset_name=None, local_full_path=None, columnsNames = None,reload=False, verbose=1):
        """Initialize local dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        self.data_keys = None
        self.columnsNames = columnsNames
        self.dataset_name = dataset_name
        self.data_directory = dataset_name
        self.file_name = dataset_name+os.path.splitext(local_full_path)[-1]
        self.local_full_path = local_full_path
        # Use default base class init()
        super(LocalDataset, self).__init__(reload=reload, verbose=verbose)

    def prepare_data(self, overwrite: bool = False, verbose: int = 1, **kwargs):
        r"""Pre-computation of molecular structure.

        Args:
            overwrite (bool): Overwrite existing database mol-json file. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        mol_filename = self.mol_filename
        if os.path.exists(os.path.join(self.data_main_dir, self.data_directory, mol_filename)) and not overwrite:
            if verbose > 0:
                print("INFO:kcnn: Found rdkit mol.json of pre-computed structures.")
            return
        filepath = os.path.join(self.data_main_dir, self.data_directory, self.file_name)
        data = pd.read_csv(filepath)
        smiles = data['smiles'].values
        mb = self._smiles_to_mol_list(smiles, add_hydrogen=True, sanitize=True, make_conformers=True, verbose=verbose)
        save_json_file(mb, os.path.join(self.data_main_dir, self.data_directory, mol_filename))

    def read_in_memory(self, has_conformers: bool = True, add_hydrogen: bool = True, verbose: int = 1):
        r"""Load ESOL data into memory and split into items. Calls :obj:`read_in_memory` of base class.

        Args:
            has_conformers (bool): If molecules have 3D coordinates pre-computed.
            add_hydrogen (bool): Whether to add H after smile translation.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        filepath = os.path.join(self.data_main_dir, self.data_directory, self.file_name)
        print('filepath:',filepath)
        data = pd.read_csv(filepath)
        # self.data_full = data
        self.data_keys = data.columns
        if verbose:
            print('file keys:',self.data_keys)
        self.graph_labels = np.expand_dims(np.array(data[self.columnsNames]), axis=-1)
        self.length = len(self.graph_labels)
        super(LocalDataset, self).read_in_memory(has_conformers=has_conformers, verbose=verbose)


# ed = ESOLDataset()


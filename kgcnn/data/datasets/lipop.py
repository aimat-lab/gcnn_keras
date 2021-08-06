import os
import numpy as np
import pandas as pd

from kgcnn.data.moleculenet import MuleculeNetDataset
from kgcnn.utils.data import save_json_file


class LipopDataset(MuleculeNetDataset):
    """Store and process full ESOL dataset."""

    dataset_name = "Lipop"
    data_main_dir = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets")
    data_directory = "Lipop"
    download_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
    file_name = 'Lipophilicity.csv'
    unpack_tar = False
    unpack_zip = False
    unpack_directory = None
    fits_in_memory = True
    require_prepare_data = True

    def __init__(self, reload=False, verbose=1):
        """Initialize ESOL dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        super(LipopDataset, self).__init__(reload=reload, verbose=verbose)

    def prepare_data(self, overwrite=False, verbose=1, **kwargs):
        mol_filename = self.mol_filename
        if os.path.exists(os.path.join(self.data_main_dir, self.data_directory, mol_filename)) and not overwrite:
            if verbose > 0:
                print("INFO:kcnn: Found rdkit mol.json of pre-computed structures.")
            return
        filepath = os.path.join(self.data_main_dir, self.data_directory, self.file_name)
        data = pd.read_csv(filepath)
        smiles = data['smiles'].values
        mb = self._smiles_to_mol_list(smiles, addHs=True, sanitize=True, embed_molecule=True, verbose=verbose)
        save_json_file(mb, os.path.join(self.data_main_dir, self.data_directory, mol_filename))

    def read_in_memory(self, verbose=1):
        filepath = os.path.join(self.data_main_dir, self.data_directory, self.file_name)
        data = pd.read_csv(filepath)
        labels2 = np.expand_dims(np.array(data['exp']), axis=-1)
        # labels1 = np.expand_dims(np.array(data['ESOL predicted log solubility in mols per litre']), axis=-1)
        self.graph_labels = labels2
        self.length = len(labels2)
        super(LipopDataset, self).read_in_memory(verbose=verbose)

ld = LipopDataset(reload=False)
# ld.define_attributes()
# labels, nodes, edges, edge_indices, graph_state = ld.get_graph()

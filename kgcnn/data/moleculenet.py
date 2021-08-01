import os
import pandas as pd
import rdkit
import rdkit.Chem

from kgcnn.data.base import DownloadDatasetBase, MemoryGraphDatasetBase
from kgcnn.mol.molgraph import MolecularGraph
from kgcnn.utils.data import load_json_file


class MuleculeNetDataset(DownloadDatasetBase, MemoryGraphDatasetBase):

    mol_filename = "mol.json"

    def __init__(self, reload=False, verbose=1):


        DownloadDatasetBase.__init__(self, reload=reload, verbose=verbose)
        MemoryGraphDatasetBase.__init__(self, verbose=verbose)

    @classmethod
    def _smiles_to_mol_list(cls, smiles, add_Hs=False, sanitize=True, embed_molecule=True):

        molecule_list = []
        for i, sm in enumerate(smiles):
            mg = MolecularGraph()
            mg.mol_from_smiles(sm, add_Hs=add_Hs, sanitize=sanitize, embed_molecule=embed_molecule)
            molecule_list.append(rdkit.Chem.MolToMolBlock(mg.mol))

        return molecule_list

    def read_in_memory(self, verbose=1):
        """Load ESOL data into memory and already split into items.

        Args:
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        mol_path = os.path.join(self.data_main_dir, self.data_directory, self.mol_filename)
        if not os.path.exists(mol_path):
            print("ERROR:kgcnn: Can not load molecules for dataset %s" % self.dataset_name)
        else:
            mols = load_json_file(mol_path)
            for x in mols:
                mg = MolecularGraph()
                mg.mol_from_molblock(x, removeHs=False, sanitize=False)

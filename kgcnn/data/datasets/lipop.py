import os
import numpy as np
import pandas as pd

from kgcnn.data.moleculenet import MoleculeNetDataset
from kgcnn.data.base import DownloadDataset
from kgcnn.utils.data import save_json_file


class LipopDataset(MoleculeNetDataset, DownloadDataset):
    """Store and process full ESOL dataset."""

    dataset_name = "Lipop"
    data_main_dir = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets")
    data_directory_name = "Lipop"
    download_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
    download_file_name = 'Lipophilicity.csv'
    unpack_tar = False
    unpack_zip = False
    unpack_directory_name = None
    fits_in_memory = True
    require_prepare_data = True

    def __init__(self, reload=False, verbose=1):
        r"""Initialize ESOL dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        self.data_keys = None
        # Use default base class init()
        MoleculeNetDataset.__init__(self, verbose=verbose)
        DownloadDataset.__init__(self, reload=reload, verbose=verbose)

        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name)
        self.file_name = self.download_file_name

        if self.require_prepare_data:
            self.prepare_data(overwrite=reload, verbose=verbose)
        if self.mol_filename:
            self.read_in_memory(verbose=verbose)

    def prepare_data(self, file_name: str = None, data_directory: str = None, dataset_name: str = None,
                     overwrite: bool = False, verbose: int = 1, smiles_column_name: str = "smiles",
                     add_hydrogen: bool = True, make_conformers: bool = True, **kwargs):
        r"""Pre-computation of molecular structure.

        Args:
            file_name (str): Filename for reading into memory. Default is None.
            data_directory (str): Full path to directory containing all files. Default is None.
            dataset_name (str): Name of the dataset. Default is None.
            overwrite (bool): Overwrite existing database mol-json file. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
            smiles_column_name (str): Column name where smiles are given in csv-file. Default is "smiles".
            add_hydrogen (bool): Whether to add H after smile translation. Default is True.
            make_conformers (bool): Whether to make conformers. Default is True.

        Returns:
            self
        """
        return super(LipopDataset, self).prepare_data(file_name=file_name, data_directory=data_directory,
                                                      dataset_name=dataset_name, overwrite=overwrite,
                                                      smiles_column_name=smiles_column_name, add_hydrogen=add_hydrogen,
                                                      verbose=verbose)

    def read_in_memory(self, file_name: str = None, data_directory: str = None, dataset_name: str = None,
                       has_conformers: bool = True,
                       label_column_name: str = 'exp',
                       add_hydrogen: bool = True, verbose: int = 1):
        r"""Load Lipop data into memory and split into items. Calls :obj:`read_in_memory` of base class.

        Args:
            file_name (str): Filename for reading into memory. Default is None.
            data_directory (str): Full path to directory containing all files. Default is None.
            dataset_name (str): Name of the dataset. Default is None.
            has_conformers (bool): If molecules have 3D coordinates pre-computed.
            label_column_name (str): Column name where labels are given in csv-file. Default is None.
            add_hydrogen (bool): Whether to add H after smile translation.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.

        Returns:
            self
        """
        return super(LipopDataset, self).read_in_memory(file_name=file_name, data_directory=data_directory,
                                                        dataset_name=dataset_name, has_conformers=has_conformers,
                                                        label_column_name=label_column_name, add_hydrogen=add_hydrogen,
                                                        verbose=verbose)

ld = LipopDataset(reload=True)
# ld.define_attributes()
# labels, nodes, edges, edge_indices, graph_state = ld.get_graph()

import os
import pickle
import numpy as np
import scipy.io
import json

from kgcnn.data.qm import QMDataset
from kgcnn.data.download import DownloadDataset
from kgcnn.mol.io import write_list_to_xyz_file
from kgcnn.graph.geom import coulomb_matrix_to_inverse_distance_proton


class QM7bDataset(QMDataset, DownloadDataset):
    """Store and process QM7b dataset."""

    download_info = {
        "dataset_name": "QM7b",
        "data_directory_name": "qm7b",
        # https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7b.mat
        "download_url": "http://quantum-machine.org/data/qm7b.mat",
        "download_file_name": 'qm7b.mat',
        "unpack_tar": False,
        "unpack_zip": False,
    }

    def __init__(self, reload: bool = False, verbose: int = 1):
        """Initialize QM9 dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        QMDataset.__init__(self, verbose=verbose, dataset_name="QM7b")
        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)

        self.label_names = ["aepbe0", "zindo-excitation-energy-with-the-most-absorption",
                            "zindo-highest-absorption", "zindo-homo", "zindo-lumo",
                            "zindo-1st-excitation-energy", "zindo-ionization-potential", "zindo-electron-affinity",
                            "ks-homo", "ks-lumo", "gw-homo", "gw-lumo", "polarizability-pbe", "polarizability-scs"]
        self.label_units = ["[?]"]*14
        self.label_unit_conversion = np.array(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 27.2114, 27.2114, 27.2114, 1.0, 27.2114, 27.2114, 27.2114,
              27.2114, 27.2114, 1.0]]
        )  # Pick always same units for training
        self.dataset_name = "QM7b"
        self.require_prepare_data = True
        self.fits_in_memory = False
        self.verbose = verbose
        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name)
        self.file_name = "qm7b.xyz"

        if self.require_prepare_data:
            self.prepare_data(overwrite=reload)

        if self.fits_in_memory:
            self.read_in_memory()

    def prepare_data(self, overwrite: bool = False, xyz_column_name: str = None, make_sdf: bool = True):
        if not os.path.exists(os.path.join(self.data_directory, self.file_name)):
            mat = scipy.io.loadmat(os.path.join(self.data_directory, self.download_info["download_file_name"]))
            labels = mat["T"]
            coulomb_mat = mat["X"]
            graph_len = [np.diagonal(x) for x in coulomb_mat]


# data = QM7bDataset()
import os
import pickle
import numpy as np
import scipy.io
import json
import pandas as pd

from kgcnn.data.qm import QMDataset
from kgcnn.data.download import DownloadDataset
from kgcnn.mol.io import write_list_to_xyz_file
from kgcnn.graph.geom import coulomb_matrix_to_inverse_distance_proton, coordinates_from_distance_matrix
from kgcnn.graph.adj import invert_distance
from kgcnn.mol.methods import inverse_global_proton_dict


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
        self.label_units = ["[?]"] * 14
        self.label_unit_conversion = np.array([1.0] * 14)  # Pick always same units for training
        self.dataset_name = "QM7b"
        self.require_prepare_data = True
        self.fits_in_memory = True
        self.verbose = verbose
        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name)
        self.file_name = "qm7b.xyz"

        if self.require_prepare_data:
            self.prepare_data(overwrite=reload)

        if self.fits_in_memory:
            self.read_in_memory(label_column_name=self.label_names)

    def prepare_data(self, overwrite: bool = False, xyz_column_name: str = None, make_sdf: bool = True):
        if not os.path.exists(os.path.join(self.data_directory, self.file_name)) or overwrite:
            mat = scipy.io.loadmat(os.path.join(self.data_directory, self.download_info["download_file_name"]))
            coulomb_mat = mat["X"]
            graph_len = [int(np.around(np.sum(np.diagonal(x) > 0))) for x in coulomb_mat]
            proton_inv_dist = [coulomb_matrix_to_inverse_distance_proton(x[:i, :i], unit_conversion=0.529177210903) for
                               x, i in zip(coulomb_mat, graph_len)]
            proton = [x[1] for x in proton_inv_dist]
            inv_dist = [x[0] for x in proton_inv_dist]
            dist = [invert_distance(x) for x in inv_dist]
            pos = [coordinates_from_distance_matrix(x) for x in dist]
            atoms = [[inverse_global_proton_dict[i] for i in x] for x in proton]
            atoms_pos = [[x, y] for x, y in zip(atoms, pos)]
            self.info("Writing XYZ file from coulomb matrix information.")
            write_list_to_xyz_file(os.path.join(self.data_directory, "qm7b.xyz"), atoms_pos)
        else:
            self.info("Found XYZ file for qm7b already created.")

        file_path = os.path.join(self.data_directory, os.path.splitext(self.file_name)[0] + ".csv")
        if not os.path.exists(file_path) or overwrite:
            mat = scipy.io.loadmat(os.path.join(self.data_directory, self.download_info["download_file_name"]))
            labels = mat["T"]
            targets = pd.DataFrame(labels, columns=self.label_names)
            self.info("Writing CSV file of graph labels.")
            targets.to_csv(file_path, index=False)
        else:
            self.info("Found CSV file of graph labels.")

        return super(QM7bDataset, self).prepare_data(
            overwrite=overwrite, xyz_column_name=xyz_column_name, make_sdf=make_sdf)

# data = QM7bDataset(reload=True)

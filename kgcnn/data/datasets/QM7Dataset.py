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


class QM7Dataset(QMDataset, DownloadDataset):
    """Store and process QM7b dataset."""

    download_info = {
        "dataset_name": "QM7",
        "data_directory_name": "qm7",
        # https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.mat
        "download_url": "http://quantum-machine.org/data/qm7.mat",
        "download_file_name": 'qm7.mat',
        "unpack_tar": False,
        "unpack_zip": False,
    }

    def __init__(self, reload: bool = False, verbose: int = 1):
        """Initialize QM9 dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        QMDataset.__init__(self, verbose=verbose, dataset_name="QM7")
        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)

        self.label_names = ["u0_atom"]
        self.label_units = ["kcal/mol"]
        self.label_unit_conversion = np.array([1.0] * 14)  # Pick always same units for training
        self.dataset_name = "QM7"
        self.require_prepare_data = True
        self.fits_in_memory = True
        self.verbose = verbose
        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name)
        self.file_name = "qm7.xyz"

        if self.require_prepare_data:
            self.prepare_data(overwrite=reload)

        if self.fits_in_memory:
            self.read_in_memory(label_column_name=self.label_names)

    def prepare_data(self, overwrite: bool = False, xyz_column_name: str = None, make_sdf: bool = True):
        file_path = os.path.join(self.data_directory, self.file_name)
        if not os.path.exists(file_path) or overwrite:
            mat = scipy.io.loadmat(os.path.join(self.data_directory, self.download_info["download_file_name"]))
            graph_len = [int(np.around(np.sum(x > 0))) for x in mat["Z"]]
            proton = [x[:i] for i, x in zip(graph_len, mat["Z"])]
            atoms = [[inverse_global_proton_dict[i] for i in x] for x in proton]
            pos = [x[:i, :]*0.529177210903 for i, x in zip(graph_len, mat["R"])]
            atoms_pos = [[x, y] for x, y in zip(atoms, pos)]
            np.save(os.path.join(self.data_directory, "qm7_splits.npy"), mat["P"])
            self.info("Writing XYZ file from coulomb matrix information.")
            write_list_to_xyz_file(file_path, atoms_pos)
        else:
            self.info("Found XYZ file for qm7b already created.")

        file_path = os.path.join(self.data_directory, os.path.splitext(self.file_name)[0] + ".csv")
        if not os.path.exists(file_path) or overwrite:
            mat = scipy.io.loadmat(os.path.join(self.data_directory, self.download_info["download_file_name"]))
            labels = mat["T"][0]
            targets = pd.DataFrame(labels, columns=self.label_names)
            self.info("Writing CSV file of graph labels.")
            targets.to_csv(file_path, index=False)
        else:
            self.info("Found CSV file of graph labels.")

        return super(QM7Dataset, self).prepare_data(
            overwrite=overwrite, xyz_column_name=xyz_column_name, make_sdf=make_sdf)

    def read_in_memory_sdf(self, **kwargs):
        super(QM7Dataset, self).read_in_memory_sdf()

        # Mean molecular weight mmw
        mass_dict = {'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984, 'S': 32.065}

        def mmw(atoms):
            mass = [mass_dict[x] for x in atoms]
            return np.array([np.mean(mass), len(mass)])

        self.assign_property("graph_attributes", [mmw(x) for x in self.obtain_property("node_symbol")])

# data = QM7Dataset(reload=False)

import os
import numpy as np
import pandas as pd
from kgcnn.data.base import MemoryGraphDataset
from kgcnn.data.download import DownloadDataset


class MD17RevisedDataset(DownloadDataset, MemoryGraphDataset):
    r"""Store and process trajectories from :obj:`MD17DatasetRevised` dataset.

    The information of the readme file is given below:

    The molecules are taken from the original MD17 dataset by
    `Chmiela et al. <https://www.science.org/doi/10.1126/sciadv.1603015>`__ , and 100,000 structures are taken,
    and the energies and forces are recalculated at the PBE/def2-SVP level of theory using very tight SCF convergence
    and very dense DFT integration grid. As such, the dataset is practically free from nummerical noise.

    One warning: As the structures are taken from a molecular dynamics simulation
    (i.e. time series data), they are not guaranteed to be independent samples.
    This is easily evident from the auto-correlation function for the original MD17 dataset

    In short: DO NOT train a model on more than 1000 samples from this dataset.
    Data already published with 50K samples on the original MD17 dataset should be considered
    meaningless due to this fact and due to the noise in the original data.

    The data:
    The ten molecules are saved in Numpy .npz format.
    The keys correspond to:

        - 'nuclear_charges'   : The nuclear charges for the molecule
        - 'coords'            : The coordinates for each conformation (in units of Angstrom)
        - 'energies'          : The total energy of each conformation (in units of kcal/mol)
        - 'forces'            : The cartesian forces of each conformation (in units of kcal/mol/Angstrom)
        - 'old_indices'       : The index of each conformation in the original MD17 dataset
        - 'old_energies'      : The energy of each conformation taken from the original MD17 dataset
        - 'old_forces'        : The forces of each conformation taken from the original MD17 dataset

    Note that for Azobenzene, only 99988 samples are available due to 11 failed DFT calculations due to van der
    Walls clash, and the original dataset only contained 99999 structures.

    Data splits:
    Five training and test splits are saved in CSV format containing the corresponding indices.

    References:

        (1) Anders Christensen, O. Anatole von Lilienfeld, Revised MD17 dataset, Materials Cloud Archive 2020.82 (2020),
            doi: 10.24435/materialscloud:wy-kn.

    """

    download_info = {
        "dataset_name": "MD17Revised",
        "data_directory_name": "MD17Revised",
        "download_url": "https://archive.materialscloud.org/record/file?filename=rmd17.tar.bz2&record_id=466",
        "download_file_name": 'rmd17.tar.bz2',
        "unpack_tar": True,
        "unpack_zip": False,
        "unpack_directory_name": "rmd17"
    }
    possible_trajectory_names = ["aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde", "naphthalene",
                                 "paracetamol", "salicylic", "toluene", "uracil"]

    def __init__(self, trajectory_name: str = None, reload=False, verbose=10):
        """Initialize MD17DatasetRevised dataset for a specific trajectory.

        Args:
            trajectory_name (str): Name of trajectory to load.
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        self.data_keys = None
        self.trajectory_name = trajectory_name
        if trajectory_name not in self.possible_trajectory_names:
            raise ValueError(
                "Name for trajectory '%s' not found. Choose: %s." % (trajectory_name, self.possible_trajectory_names))

        MemoryGraphDataset.__init__(self, dataset_name="MD17Revised", verbose=verbose)
        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)
        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name,
                                           self.unpack_directory_name, "rmd17")
        self.file_name = "rmd17_%s.npz" % self.trajectory_name
        self.dataset_name = self.dataset_name + "_" + self.trajectory_name
        # May add name of trajectory to name of dataset here.

        if self.fits_in_memory:
            self.read_in_memory()

    def _get_trajectory_from_npz(self, file_path: str = None):
        if file_path is None:
            file_dir = os.path.join(self.data_directory, "npz_data")
            file_path = os.path.join(file_dir, self.file_name)
        return np.load(file_path)

    def _get_train_test_splits(self):
        file_dir = os.path.join(self.data_directory, "splits")

        def read_splits(file_name: str) -> list:
            return [
                np.squeeze(pd.read_csv(
                    os.path.join(file_dir, file_name % i), header=None).values, axis=-1) for i in range(1, 6)]
        return read_splits("index_train_0%i.csv"), read_splits("index_test_0%i.csv")

    def read_in_memory(self,):
        """Read dataset trajectory into memory.

        Returns:
            self.
        """
        data = self._get_trajectory_from_npz()
        self.data_keys = list(data.keys())
        for key in ["coords", "energies", "forces", "old_indices", "old_energies", "old_forces"]:
            self.assign_property(key, [x for x in data[key]])
        node_number = data["nuclear_charges"]
        self.assign_property("nuclear_charges", [np.array(node_number) for _ in range(len(self))])

        # Add splits to self.
        splits_train, splits_test = self._get_train_test_splits()
        property_train = []
        property_test = []
        for i in range(len(self)):
            is_train = []
            is_test = []
            for j, split in enumerate(splits_train):
                if i in split:
                    is_train.append(j + 1)
            for j, split in enumerate(splits_test):
                if i in split:
                    is_test.append(j + 1)
            # Add to list
            if len(is_train) > 0:
                property_train.append(np.array(is_train, dtype="int"))
            else:
                property_train.append(None)
            if len(is_test) > 0:
                property_test.append(np.array(is_test, dtype="int"))
            else:
                property_test.append(None)
        self.assign_property("train", property_train)
        self.assign_property("test", property_test)

        return self

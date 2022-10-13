import os

import numpy as np
from kgcnn.data.base import MemoryGraphDataset
from kgcnn.data.download import DownloadDataset


class MD17DatasetRevised(DownloadDataset, MemoryGraphDataset):
    """Store and process full MD17DatasetRevised dataset."""

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
        return self

# ds = MD17DatasetRevised("aspirin")

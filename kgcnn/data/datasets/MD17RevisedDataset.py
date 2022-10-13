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

    def __init__(self, trajectory_name: str = None, reload=False, verbose=1):
        """Initialize full Cora dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Use default base class init()
        self.data_keys = None
        self.trajectory_name = trajectory_name
        if trajectory_name not in self.possible_trajectory_names:
            raise ValueError(
                "Name for trajectory '%s' not found. Choose: %s." % (trajectory_name, self.possible_trajectory_names))

        MemoryGraphDataset.__init__(self, dataset_name="MD17Revised", verbose=verbose)
        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)
        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name,
                                           self.unpack_directory_name, "rmd17")

        if self.fits_in_memory:
            self.read_in_memory(self.trajectory_name)

    def read_in_memory(self, trajectory_name: str):
        """Load full Cora data into memory and already split into items."""
        file_dir = os.path.join(self.data_directory, "npz_data")
        file_path = os.path.join(file_dir, "rmd17_%s.npz" % trajectory_name)
        data = np.load(file_path)
        self.data_keys = list(data.keys())

        return data

ds = MD17DatasetRevised("aspirin")

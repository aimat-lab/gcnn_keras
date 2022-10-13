import os

import numpy as np
from kgcnn.data.base import MemoryGraphDataset
from kgcnn.data.download import DownloadDataset


class MD17Dataset(DownloadDataset, MemoryGraphDataset):
    """Store and process full MD17Dataset dataset."""
    datasets_download_info = {
        "CG-CG": {"download_file_name": "CG-CG.npz"},
        "aspirin_dft": {"download_file_name": "aspirin_dft.npz"},
        "azobenzene_dft": {"download_file_name": "azobenzene_dft.npz"},
        "benzene2017_dft": {"download_file_name": "benzene2017_dft.npz"},
        "benzene2018_dft": {"download_file_name": "benzene2018_dft.npz"},
        "ethanol_dft": {"download_file_name": "ethanol_dft.npz"},
        "malonaldehyde_dft": {"download_file_name": "malonaldehyde_dft.npz"},
        "naphthalene_dft": {"download_file_name": "naphthalene_dft.npz"},
        "paracetamol_dft": {"download_file_name": "paracetamol_dft.npz"},
        "salicylic_dft": {"download_file_name": "salicylic_dft.npz"},
        "toluene_dft": {"download_file_name": "toluene_dft.npz"},
        "uracil_dft": {"download_file_name": "toluene_dft.npz"}
    }

    def __init__(self, trajectory_name: str = None, reload=False, verbose=10):
        """Initialize full Cora dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        self.data_keys = None
        self.trajectory_name = trajectory_name
        MemoryGraphDataset.__init__(self, dataset_name="MD17", verbose=verbose)

        # Prepare download
        if trajectory_name in self.datasets_download_info:
            self.download_info = self.datasets_download_info[trajectory_name]
            self.download_info.update({
                "download_url": "http://quantum-machine.org/gdml/data/npz/%s" % self.download_info[
                    "download_file_name"]})
        else:
            raise ValueError(
                "Can not resolve '%s' trajectory. Choose: %s." % (
                    trajectory_name, list(self.datasets_download_info.keys())))

        DownloadDataset.__init__(self, dataset_name="MD17", data_directory_name="MD17", **self.download_info,
                                 reload=reload, verbose=verbose)
        self.file_name = self.download_file_name
        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name)
        self.dataset_name = self.dataset_name + "_" + self.trajectory_name
        if self.fits_in_memory:
            self.read_in_memory()

    def _get_trajectory_from_npz(self, file_path: str = None):
        if file_path is None:
            file_dir = os.path.join(self.data_directory)
            file_path = os.path.join(file_dir, self.file_name)
        return np.load(file_path)

    def read_in_memory(self):
        data = self._get_trajectory_from_npz()
        self.data_keys = list(data.keys())
        for key in ["R", "E", "F"]:
            self.assign_property(key, [x for x in data[key]])
        for key in ["z", 'name', 'type', 'md5', "theory"]:
            value = data[key]
            self.assign_property(key, [np.array(value) for _ in range(len(self))])
        return self

# ds = MD17Dataset("aspirin_dft")

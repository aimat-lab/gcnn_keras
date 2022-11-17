import os
from ase.db import connect

from kgcnn.data.base import MemoryGraphDataset
from kgcnn.data.download import DownloadDataset


class ISO17Dataset(DownloadDataset, MemoryGraphDataset):
    """Store and process full dataset."""

    download_info = {
        "dataset_name": "ISO17",
        "data_directory_name": "ISO17",
        "download_url": "http://quantum-machine.org/datasets/iso17.tar.gz",
        "download_file_name": 'iso17.tar.gz',
        "unpack_tar": True,
        "unpack_zip": False,
        "unpack_directory_name": "iso17"
    }

    def __init__(self, reload=False, verbose=1):
        """Initialize full Cora dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Use default base class init()
        self.data_keys = None

        MemoryGraphDataset.__init__(self, dataset_name="ISO17", verbose=verbose)
        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)

        self.data_directory = os.path.join(
            self.data_main_dir, self.data_directory_name, self.unpack_directory_name, "iso17")

        if self.fits_in_memory:
            self.read_in_memory()

    def read_in_memory(self):
        """Load full Cora data into memory and already split into items."""

        with connect(os.path.join(self.data_directory, "reference.db")) as conn:
            for row in conn.select(limit=10):
                print(row.numbers)
                print(row.positions)
                print(row.symbols)
                print(row['total_energy'])
                print(row.data['atomic_forces'])

        return self

ds = ISO17Dataset()

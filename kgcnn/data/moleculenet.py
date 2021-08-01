import pandas as pd
import os

from kgcnn.data.base import DownloadDatasetBase


class MuleculeNetDataset(DownloadDatasetBase):

    def read_in_memory(self, verbose=1):
        """Load ESOL data into memory and already split into items.

        Args:
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        filepath = os.path.join(self.data_main_dir, self.data_directory, self.file_name)
        data = pd.read_csv(filepath)
        self.data = data
        self.data_keys = data.columns


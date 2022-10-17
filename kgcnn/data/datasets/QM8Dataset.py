import os
import pickle
import numpy as np
import scipy.io
import json
import pandas as pd
from typing import Union
from kgcnn.data.qm import QMDataset
from kgcnn.data.download import DownloadDataset


class QM8Dataset(QMDataset, DownloadDataset):
    """Store and process QM8 dataset."""

    download_info = {
        "dataset_name": "QM8",
        "data_directory_name": "qm8",
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb8.tar.gz",
        "download_file_name": 'gdb8.tar.gz',
        "unpack_tar": True,
        "unpack_zip": False,
        "unpack_directory_name": "gdb8"
    }

    def __init__(self, reload: bool = False, verbose: int = 1):
        """Initialize QM8 dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        QMDataset.__init__(self, verbose=verbose, dataset_name="QM8")
        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)

        self.label_names = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0","f1-PBE0",
            "f2-PBE0","E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM"
        ]
        self.label_units = ["[?]"]*len(self.label_names)
        self.label_unit_conversion = np.array([1.0] * 14)  # Pick always same units for training
        self.dataset_name = "QM8"
        self.require_prepare_data = False
        self.fits_in_memory = True
        self.verbose = verbose
        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name, self.unpack_directory_name)
        self.file_name = "qm8.csv"

        if not os.path.exists(self.file_path):
            original_name = os.path.join(self.data_directory, "qm8.sdf.csv")
            if os.path.exists(original_name):
                os.rename(original_name, self.file_path)

        if self.require_prepare_data:
            self.prepare_data(overwrite=reload)

        if self.fits_in_memory:
            self.read_in_memory(label_column_name=self.label_names)


# ds = QM8Dataset(reload=True)
import os
import numpy as np

from kgcnn.data.datasets.tudataset2020 import GraphTUDataset2020
from kgcnn.mol.molgraph import OneHotEncoder


class PROTEINSDatset(GraphTUDataset2020):
    """Store and process PROTEINS dataset."""

    def __init__(self, reload=False, verbose=1):
        """Initialize MUTAG dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Use default base class init()
        # We set dataset_name to None since all flags are defined by hand in subclass definition.
        super(PROTEINSDatset, self).__init__(dataset_name="PROTEINS", reload=reload, verbose=verbose)

    def read_in_memory(self, file_name: str = None, data_directory: str = None, dataset_name: str = None,
                       verbose: int = 1):
        r"""Load PROTEINS Dataset into memory and already split into items with further cleaning and
        processing.

        Args:
            file_name (str): Filename for reading into memory. Not used for general TUDataset.
                Only for download of class `tudataset2020`. Default is None.
            data_directory (str): Full path to directory containing all txt-files. Default is None.
            dataset_name (str): Name of the dataset. Not used for reading. Default is None.
            verbose (int): Print progress or info for processing, where 0 is silent. Default is 1.
        """
        super(PROTEINSDatset, self).read_in_memory(file_name=file_name, data_directory=data_directory,
                                                   dataset_name=dataset_name, verbose=verbose)

        self.graph_labels = np.array([[0, 1] if int(x) == 2 else [1, 0] for x in self.graph_labels])
        ohe = OneHotEncoder(
            [-538, -345, -344, -134, -125, -96, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 47, 61, 63, 73, 74, 75,
             82, 104, 353, 355, 360, 558, 797, 798], add_unknown=False)
        self.node_attributes = [np.array([ohe(int(y)) for y in x]) for x in self.node_attributes]
        ohe2 = OneHotEncoder([0, 1, 2], add_unknown=False)
        self.node_labels = [np.array([ohe2(int(y)) for y in x]) for x in self.node_labels]
        ohe3 = OneHotEncoder([i for i in range(0, 17)], add_unknown=False)
        self.node_degree = [np.array([ohe3(int(y)) for y in x]) for x in self.node_degree]
        self.length = len(self.graph_labels)
        self.graph_attributes = None
        self.graph_size = [len(x) for x in self.node_attributes]

        return self

# ds = PROTEINSDatset()

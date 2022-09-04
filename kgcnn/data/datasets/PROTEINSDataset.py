import os
import numpy as np

from kgcnn.data.datasets.GraphTUDataset2020 import GraphTUDataset2020
from kgcnn.mol.encoder import OneHotEncoder


class PROTEINSDataset(GraphTUDataset2020):
    """Store and process PROTEINS dataset."""

    def __init__(self, reload=False, verbose=1):
        """Initialize MUTAG dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Use default base class init()
        # We set dataset_name to None since all flags are defined by hand in subclass definition.
        super(PROTEINSDataset, self).__init__(dataset_name="PROTEINS", reload=reload, verbose=verbose)

    def read_in_memory(self):
        r"""Load PROTEINS Dataset into memory and already split into items with further cleaning and
        processing.
        """
        super(PROTEINSDataset, self).read_in_memory()
        # One-hot encoders
        ohe = OneHotEncoder(
            [-538, -345, -344, -134, -125, -96, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 47, 61, 63, 73, 74, 75,
             82, 104, 353, 355, 360, 558, 797, 798], add_unknown=False)
        ohe2 = OneHotEncoder([0, 1, 2], add_unknown=False)
        ohe3 = OneHotEncoder([i for i in range(0, 17)], add_unknown=False)

        graph_labels = self.obtain_property("graph_labels")
        node_attributes = self.obtain_property("node_attributes")
        node_labels = self.obtain_property("node_labels")
        node_degree = self.obtain_property("node_degree")
        self.assign_property("graph_labels", [x - 1 for x in graph_labels])
        self.assign_property("node_attributes", [np.array([ohe(int(y)) for y in x]) for x in node_attributes])
        self.assign_property("node_labels", [np.array([ohe2(int(y)) for y in x]) for x in node_labels])
        self.assign_property("node_degree", [np.array([ohe3(int(y)) for y in x]) for x in node_degree])
        self.assign_property("graph_size", [len(x) if x is not None else None for x in node_attributes])

        return self

# ds = PROTEINSDataset()

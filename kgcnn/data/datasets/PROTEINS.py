import os
import numpy as np

from kgcnn.data.tudataset import GraphTUDataset
from kgcnn.mol.molgraph import OneHotEncoder


class PROTEINSDatset(GraphTUDataset):
    """Store and process PROTEINS dataset."""

    data_main_dir = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets")
    data_directory = "PROTEINS"
    download_url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/PROTEINS.zip"
    file_name = "PROTEINS.zip"
    unpack_zip = True
    unpack_directory = "PROTEINS"
    kgcnn_dataset_name = "PROTEINS"
    fits_in_memory = True

    def __init__(self, reload=False, verbose=1):
        """Initialize MUTAG dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Use default base class init()
        # We set tudataset_name to None since all flags are defined by hand in subclass definition.
        super(PROTEINSDatset, self).__init__(tudataset_name=None, reload=reload, verbose=verbose)

    def read_in_memory(self, verbose=1):
        """Load PROTEINS data into memory and already split into items.

        Args:
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        super(PROTEINSDatset, self).read_in_memory(verbose=verbose)
        self.labels_graph = np.array([[0, 1] if int(x) == 2 else [1, 0] for x in self.labels_graph])
        ohe = OneHotEncoder(
            [-538, -345, -344, -134, -125, -96, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 47, 61, 63, 73, 74, 75,
             82, 104, 353, 355, 360, 558, 797, 798], add_others=False)
        self.nodes = [np.array([ohe(int(y)) for y in x]) for x in self.nodes]
        ohe2 = OneHotEncoder([0, 1, 2], add_others=False)
        self.labels_node = [np.array([ohe2(int(y)) for y in x]) for x in self.labels_node]
        ohe3 = OneHotEncoder([i for i in range(0, 17)], add_others=False)
        self.nodes_degree = [np.array([ohe3(int(y)) for y in x]) for x in self.nodes_degree]

    def get_graph(self):
        """Make graph tensor objects for MUTAG.

        Returns:
            tuple: labels, nodes, edge_indices, edges
        """
        return self.labels_graph, self.nodes, self.edge_indices, self.edges


# ds = PROTEINSDatset()

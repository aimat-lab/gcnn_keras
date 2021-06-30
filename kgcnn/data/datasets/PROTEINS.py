import os
import numpy as np

from kgcnn.data.base import GraphDatasetBase


class PROTEINSDatset(GraphDatasetBase):
    """Store and process PROTEINS dataset."""

    data_main_dir = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets")
    data_directory = "PROTEINS"
    download_url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/PROTEINS.zip"
    file_name = "PROTEINS.zip"
    unpack_zip = True
    unpack_directory = "PROTEINS"

    def __init__(self, reload=False, verbose=1):
        """Initialize MUTAG dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Use default base class init()
        super(PROTEINSDatset, self).__init__(reload=reload, verbose=verbose)

    def read_in_memory(self, verbose=1):
        """Load MUTAG data into memory and already split into items.

        Args:
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """


    def get_graph(self):
        """Make graph tensor objects for MUTAG.

        Returns:
            tuple: labels, nodes, edge_indices, edges
        """
        return self.labels, self.nodes, self.edge_indices, self.edges


ds = PROTEINSDatset()

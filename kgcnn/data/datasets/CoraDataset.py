import os

import numpy as np
import scipy.sparse as sp

from kgcnn.data.base import MemoryGraphDataset
from kgcnn.data.download import DownloadDataset
from kgcnn.graph.adj import convert_scaled_adjacency_to_list


class CoraDataset(DownloadDataset, MemoryGraphDataset):
    """Store and process full Cora dataset."""

    download_info = {
        "dataset_name": "Cora",
        "data_directory_name": "cora",
        "download_url": "https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz",
        "download_file_name": 'cora.npz',
        "unpack_tar": False,
        "unpack_zip": False,
        "unpack_directory_name": None
    }

    def __init__(self, reload=False, verbose=1):
        """Initialize full Cora dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Use default base class init()
        self.data_keys = None

        MemoryGraphDataset.__init__(self, dataset_name="Cora", verbose=verbose)
        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)

        if self.fits_in_memory:
            self.read_in_memory()

    def read_in_memory(self):
        """Load full Cora data into memory and already split into items."""
        filepath = os.path.join(self.data_main_dir, self.data_directory_name, "cora.npz")
        loader = np.load(filepath, allow_pickle=True)
        loader = dict(loader)

        a = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        x = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        # Original adjacency matrix
        # self.graph_adjacency = a

        # Compute labels
        labels = loader.get('labels')
        labels = np.expand_dims(labels, axis=-1)
        labels = np.array(labels == np.arange(70), dtype="float")
        self.assign_property("node_labels", [labels])  # One graph

        # Node attributes
        self.assign_property("node_attributes", [x.toarray()])

        # Set edges and indices.
        edi, ed = convert_scaled_adjacency_to_list(a)
        self.assign_property("edge_indices", [edi])
        self.assign_property("edge_attributes", [np.expand_dims(ed, axis=-1)])
        self.assign_property("edge_weights", [np.expand_dims(ed, axis=-1)])

        # Information
        self.data_keys = loader.get('idx_to_class')

        return self

# ds = CoraDataset()

import os

import numpy as np
import scipy.sparse as sp

from kgcnn.data.base import DownloadDataset, MemoryGraphDataset
from kgcnn.utils.adj import convert_scaled_adjacency_to_list, add_edges_reverse_indices, precompute_adjacency_scaled, make_adjacency_undirected_logical_or


class CoraDataset(DownloadDataset, MemoryGraphDataset):
    """Store and process full Cora dataset."""

    dataset_name = "Cora"
    data_main_dir = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets")
    data_directory = "cora"
    download_url = "https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz"
    file_name = 'cora.npz'
    unpack_tar = False
    unpack_zip = False
    unpack_directory = None
    fits_in_memory = True

    def __init__(self, reload=False, verbose=1):
        """Initialize full Cora dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Use default base class init()
        self.data_keys = None
        self.length = 1

        DownloadDataset.__init__(self, reload=reload, verbose=verbose)
        MemoryGraphDataset.__init__(self, verbose=verbose)

        if self.fits_in_memory:
            self.read_in_memory(verbose=verbose)

    def read_in_memory(self, verbose=1):
        """Load full Cora data into memory and already split into items.

        Args:
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        filepath = os.path.join(self.data_main_dir, self.data_directory, "cora.npz")
        loader = np.load(filepath, allow_pickle=True)
        loader = dict(loader)

        a = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        x = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        # Original adjacency matrix
        self.graph_adjacency = a

        # Compute labels
        labels = loader.get('labels')
        labels = np.expand_dims(labels, axis=-1)
        labels = np.array(labels == np.arange(70), dtype=np.float)
        self.node_labels = [labels]  # One graph

        # Node attributes
        self.node_attributes = [x.toarray()]

        # Set edges and indices.
        edi, ed = convert_scaled_adjacency_to_list(a)
        self.edge_indices = [edi]
        self.edge_attributes = [np.expand_dims(ed,axis=-1)]

        # Information
        self.data_keys = loader.get('idx_to_class')

        return self

    def make_undirected_edges(self):
        """Make edges undirected, however leave the original adjacency matrix as-is!!"""
        self.graph_adjacency = make_adjacency_undirected_logical_or(self.graph_adjacency)
        edi, ed = add_edges_reverse_indices(self.edge_indices[0], self.edge_attributes[0])
        self.edge_indices = [edi]
        self.edge_attributes = [ed]
        return self

    def scale_adjacency(self):
        self.graph_adjacency = precompute_adjacency_scaled(self.graph_adjacency)
        edi, ed = convert_scaled_adjacency_to_list(self.graph_adjacency)
        self.edge_indices = [edi]
        self.edge_attributes = [np.expand_dims(ed, axis=-1)]
        return self



# ds = CoraDataset()
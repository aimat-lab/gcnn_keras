import os

import numpy as np
import scipy.sparse as sp

from kgcnn.data.base import GraphDatasetBase


class CoraDataset(GraphDatasetBase):
    """Store and process full Cora dataset."""

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
        super(CoraDataset, self).__init__(reload=reload, verbose=verbose)

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

        # adj_data = loader['adj_data']
        # adj_ind = loader['adj_indices']
        # adj_indptr = loader['adj_indptr']
        # adj_shape = loader['adj_shape']
        # adj_idx_list = []
        # adj_val_list = []
        # for i in range(adj_shape[0]):
        #     cols = np.expand_dims(adj_ind[adj_indptr[i]:adj_indptr[i+1]],axis=-1)
        #     rows = np.zeros_like(cols,dtype=np.int)
        #     rows[:,:] = i
        #     idxs = np.concatenate([rows,cols],axis=-1)
        #     adj_val_list.append(adj_data[adj_indptr[i]:adj_indptr[i+1]])
        #     adj_idx_list.append(idxs)
        # adj_idx_list = np.concatenate(adj_idx_list,axis=0)
        # adj_val_list = np.concatenate(adj_val_list,axis=0)

        x = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])

        # attr_data = loader['attr_data']
        # attr_ind = loader['attr_indices']
        # attr_indptr = loader['attr_indptr']
        # attr_shape = loader['attr_shape']
        # attr_idx_list = []
        # attr_val_list = []
        # for i in range(attr_shape[0]):
        #     cols = np.expand_dims(attr_ind[attr_indptr[i]:attr_indptr[i+1]],axis=0)
        #     colval = attr_data[attr_indptr[i]:attr_indptr[i+1]]
        #     colval_padded = np.zeros(attr_shape[1])
        #     colval_padded[cols] = colval
        #     attr_val_list.append(colval_padded)
        #     attr_idx_list.append(cols)
        # attr_val_list = np.array(attr_val_list)

        self.labels_node = loader.get('labels')
        self.graph_adjacency = a
        self.nodes = x

        return self.graph_adjacency, self.nodes, self.labels_node

    def get_graph(self):
        """Make graph tensor objects for Cora dataset.

        Returns:
            tuple: A, X, labels
        """
        return self.graph_adjacency, self.nodes, self.labels_node

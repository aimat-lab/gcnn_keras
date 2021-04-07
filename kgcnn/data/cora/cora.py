import os

import numpy as np
import requests
import scipy.sparse as sp

from kgcnn.utils.data import setup_user_database_directory


# import pickle
# import shutil
# import zipfile

def cora_download_dataset(path, overwrite=False):
    """
    Download Mutag as zip-file.
    
    Args:
        path: (str) filepath if empty use user-default path
        overwrite: (bool) overwrite existing database, default:False
    
    Returns:
        os.path: Filepath
    """
    if os.path.exists(os.path.join(path, 'cora.npz')) is False or overwrite:
        print("Downloading dataset...", end='', flush=True)
        data_url = "https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz"
        r = requests.get(data_url, allow_redirects=True)
        open(os.path.join(path, 'cora.npz'), 'wb').write(r.content)
        print("done")
    else:
        print("Dataset found ... done")
    return os.path.join(path, 'cora.npz')


def cora_make_graph(loader):
    """
    Load the cora dataset.

    Args:
        loader (dict): Dictionary from .npz file.

    Returns:
        list: [adj_matrix,nodes,labels]
        
        - adj_matrix (sp.csr_matrix): Adjacency matrix.
        - nodes (sp.csr_matrix): Node features.
        - labels (np.array): Labels.
    """
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

    labels = loader.get('labels')

    return a, x, labels


def cora_graph(filepath=None):
    """
    Load and convert cora citation dataset.

    Args:
        filepath (str): Path to dataset. Default is None.

    Returns:
        list: [adj_matrix,X,labels]
        
        - adj_matrix (sp.csr_matrix): Adjacency matrix.
        - X (sp.csr_matrix): Node features.
        - labels (np.array): Labels.
    """
    user_default_base = setup_user_database_directory()
    if filepath is None:
        filepath = os.path.join(str(user_default_base), "data", "cora")

    print("Database path:", filepath)
    if not os.path.exists(os.path.join(filepath, "cora.npz")):
        cora_download_dataset(filepath)

    loader = np.load(os.path.join(filepath, "cora.npz"), allow_pickle=True)
    loader = dict(loader)
    data = cora_make_graph(loader)

    return data

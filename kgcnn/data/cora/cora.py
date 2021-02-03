import pickle
import zipfile
import os
import requests
import numpy as np
import shutil
import scipy.sparse as sp


def cora_download_dataset(path,overwrite=False):
    """
    Download Mutag as zip-file.
    
    Args:
        datadir: (str) filepath if empty use user-default path
        overwrite: (bool) overwrite existing database, default:False
    
    Returns:
        os.path: Filepath
    """ 
    if(os.path.exists(os.path.join(path,'cora.npz')) == False or overwrite == True):
        print("Downloading dataset...", end='', flush=True)
        data_url = "https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz"
        r = requests.get(data_url,allow_redirects=True)
        open(os.path.join(path,'cora.npz'),'wb').write(r.content) 
        print("done")
    else:
        print("Dataset found ... done")
    return os.path.join(path,'cora.npz') 



def cora_make_grph(loader):
    """
    Load the cora dataset.

    Args:
        loader (dict): Dictionary from .npz file.

    Returns:
        list: [A,X,labels]
        
        - A (sp.csr_matrix): Adjacency matrix.
        - X (sp.csr_matrix): Node features.
        - labels (np.array): Labels.
    """   
    A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
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
    
    
    X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
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
    
    return A,X,labels


def cora_graph():
    """
    Load and convert cora citation dataset.

    Returns:
        list: [A,X,labels]
        
        - A (sp.csr_matrix): Adjacency matrix.
        - X (sp.csr_matrix): Node features.
        - labels (np.array): Labels.
    """
    local_path = os.path.split(os.path.realpath(__file__))[0]
    print("Database path:",local_path)
    if(os.path.exists(os.path.join(local_path,"cora.npz"))==False):
        cora_download_dataset(local_path)
    
    loader = np.load(os.path.join(local_path,"cora.npz"),allow_pickle=True)
    loader = dict(loader)
    data  = cora_make_grph(loader)
    
    return data

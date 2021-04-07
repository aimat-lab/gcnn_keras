import os
# import pickle
# import shutil
# import zipfile
import gzip
import tarfile
import shutil

import numpy as np
import requests

from kgcnn.utils.data import setup_user_database_directory


def cora_download_dataset(path, overwrite=False):
    """
    Download Mutag as zip-file.

    Args:
        path: (str) filepath if empty use user-default path
        overwrite: (bool) overwrite existing database, default:False

    Returns:
        os.path: Filepath
    """
    if os.path.exists(os.path.join(path, 'cora.tgz')) is False or overwrite:
        print("Downloading dataset...", end='', flush=True)
        # From https://linqs.soe.ucsc.edu/data
        data_url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
        # data_url = "https://linqs-data.soe.ucsc.edu/public/arxiv-mrdm05/arxiv.tar.gz"
        r = requests.get(data_url, allow_redirects=True)
        open(os.path.join(path, 'cora.tgz'), 'wb').write(r.content)
        print("done")
    else:
        print("Dataset found ... done")
    return os.path.join(path, 'cora.npz')


def cora_extract_dataset(path, overwrite=False):
    """
    Extract dsgdb9nsd.xyz zip-file.

    Args:
        path: (str) filepath if empty use user-default path
        overwrite: (bool) overwrite existing database, default:False

    Returns:
        os.path: Filepath
    """
    if not os.path.exists(os.path.join(path, 'cora_lu')):
        print("Creating directory ... ", end='', flush=True)
        os.mkdir(os.path.join(path, 'cora_lu'))
        print("done")
    else:
        print("Directory for extraction exists ... done")
        if not overwrite:
            print("Not extracting Zip File ... stopped")
            return os.path.join(path, 'cora_lu')

    print("Read Zip File ... ", end='', flush=True)
    # archive = tarfile.open(os.path.join(path, 'cora.tar.gz'), "r")
    # Filelistnames = archive.getnames()
    print("done")

    print("Extracting Zip folder ... ", end='', flush=True)
    # archive.extractall(os.path.join(path, 'cora_arxiv'))
    shutil.unpack_archive(os.path.join(path, 'cora.tgz'),os.path.join(path, 'cora_lu'))
    print("done")
    # archive.close()

    return os.path.join(path, 'cora_lu')


def cora_make_graph(filepath):
    """
    Make cora graph that was published by Qing Lu, and Lise Getoor. "Link-based classification." ICML, 2003.
    https://www.aaai.org/Papers/ICML/2003/ICML03-066.pdf

    Args:
        filepath (str): Path to directory.

    Returns:
        list: [nodes, edge_indices, labels, mapping]

        - nodes(np.array): Node features of shape (2708,1434)
        - edge_indices (np.array): Edge edge_indices of shape (5429,2)
        - labels (np.array): Class labels of citations of shape (2708,)
        - mapping (dict): Class label mapping.
    """
    ids = np.loadtxt(os.path.join(filepath,"cora.cites"))
    ids = np.array(ids,np.int)
    open_file = open(os.path.join(filepath, "cora.content"), "r")
    lines = open_file.readlines()
    labels = [x.strip().split('\t')[-1] for x in lines]
    nodes = [x.strip().split('\t')[0:-1] for x in lines]
    nodes = np.array([[int(y) for y in x] for x in nodes],dtype=np.int)
    open_file.close()
    # Match edge_indices not wiht ids but with edge_indices in nodes
    node_map = np.zeros(np.max(nodes[:,0])+1,dtype=np.int)
    idx_new = np.arange(len(nodes))
    node_map[nodes[:,0]] = idx_new
    indexlist = node_map[ids]
    order1 = np.argsort(indexlist[:,1],axis=0,kind='mergesort') # stable!
    ind1 = indexlist[order1]
    order2 = np.argsort(ind1[:,0],axis=0,kind='mergesort')
    indices = ind1[order2]
    # Class mappings
    class_label_mapping = {'Genetic_Algorithms': 0,
                    'Reinforcement_Learning': 1,
                    'Theory': 2,
                    'Rule_Learning': 3,
                    'Case_Based': 4,
                    'Probabilistic_Methods': 5,
                    'Neural_Networks': 6}
    label_id = np.array([class_label_mapping[x] for x in labels],dtype=np.int)
    return nodes,indices,label_id,class_label_mapping

def cora_graph(filepath=None):
    """
    Load and convert cora citation dataset.

    Args:
        filepath (str): Path to dataset. Default is None.

    Returns:
        list: [nodes, edge_indices, labels, mapping]

        - nodes(np.array): Node features of shape (2708,1434)
        - edge_indices (np.array): Edge edge_indices of shape (5429,2)
        - labels (np.array): Class labels of citations of shape (2708,)
        - mapping (dict): Class label mapping.
    """
    if filepath is None:
        user_database = setup_user_database_directory()
        filepath = os.path.join(str(user_database), "data", "cora")

    print("Database path:", filepath)
    if not os.path.exists(os.path.join(filepath, "cora.tgz")):
        cora_download_dataset(filepath)

    if not os.path.exists(os.path.join(filepath, "cora_lu")):
        cora_extract_dataset(filepath)

    data = cora_make_graph(os.path.join(filepath,"cora_lu","cora"))

    return data

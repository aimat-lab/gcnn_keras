import pickle
import tensorflow as tf
import numpy as np
import os

def save_list(outlist,fname):
    """
    Save a pickled list to file.

    Args:
        outlist (list): List of eg. np.arrays.
        fname (str): Filepath to save.

    Returns:
        None.
    """
    with open(fname,'wb') as f: 
        pickle.dump(outlist, f)


def load_list(fname):
    """
    Load a pickled list from file.

    Args:
        fname (str): Filepath to load.

    Returns:
        outlist (list): Pickle object.
    """
    with open(fname,'rb') as f: 
        outlist = pickle.load(f)
    return outlist



def ragged_tensor_from_nested_numpy(numpy_list):
    """
    Make ragged tensor from a list of numpy arrays. Ragged dimension only as first axis.

    Args:
        numpy_list (list): List of numpy arrays. Example [np.array, np.array, ...]

    Returns:
        tf.raggedTensor
    """
    return tf.RaggedTensor.from_row_lengths(np.concatenate(numpy_list,axis=0), np.array([len(x) for x in numpy_list],dtype=np.int))


def setup_user_database_directory():
    """
    Setup directory for graph databases at '~/.kgcnn'.

    Returns:
        os.path: Path to data directory in user folder.
    """
    if not os.path.exists(os.path.join(os.path.expanduser("~"), ".kgcnn")):
        print("Setup local data folder for kgcnn at: ", os.path.join(os.path.expanduser("~"), ".kgcnn"))
        os.mkdir(os.path.join(os.path.expanduser("~"), ".kgcnn"))
    if not os.path.exists(os.path.join(os.path.expanduser("~"), ".kgcnn","data")):
        os.mkdir(os.path.join(os.path.expanduser("~"), ".kgcnn","data"))

    # Make individual data directories
    if not os.path.exists(os.path.join(os.path.expanduser("~"), ".kgcnn","data","qm")):
        os.mkdir(os.path.join(os.path.expanduser("~"), ".kgcnn","data","qm"))
    if not os.path.exists(os.path.join(os.path.expanduser("~"), ".kgcnn","data","mutagen")):
        os.mkdir(os.path.join(os.path.expanduser("~"), ".kgcnn","data","mutagen"))
    if not os.path.exists(os.path.join(os.path.expanduser("~"), ".kgcnn","data","cora")):
        os.mkdir(os.path.join(os.path.expanduser("~"), ".kgcnn","data","cora"))

    return os.path.join(os.path.expanduser("~"), ".kgcnn")
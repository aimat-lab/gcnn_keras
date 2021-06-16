import pickle
import tensorflow as tf
import numpy as np
# import os


def save_pickle_file(outlist, fname):
    """Save a pickled list to file.

    Args:
        outlist (list): List of eg. np.arrays.
        fname (str): Filepath to save.

    Returns:
        None.
    """
    with open(fname, 'wb') as f:
        pickle.dump(outlist, f)


def load_pickle_file(fname):
    """Load a pickled list from file.

    Args:
        fname (str): Filepath to load.

    Returns:
        pickle: Pickled object.
    """
    with open(fname, 'rb') as f:
        outlist = pickle.load(f)
    return outlist


def ragged_tensor_from_nested_numpy(numpy_list):
    """Make ragged tensor from a list of numpy arrays. Ragged dimension only as first axis.

    Args:
        numpy_list (list): List of numpy arrays. Example [np.array, np.array, ...]

    Returns:
        tf.RaggedTensor: Ragged tensor of former nested list of numpy arrays.
    """
    return tf.RaggedTensor.from_row_lengths(np.concatenate(numpy_list, axis=0), np.array([len(x) for x in numpy_list],
                                                                                         dtype=np.int))

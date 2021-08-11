import pickle
import tensorflow as tf
import numpy as np
import yaml
import json
# import os


def save_pickle_file(outlist, fname):
    with open(fname, 'wb') as f:
        pickle.dump(outlist, f)


def load_pickle_file(fname):
    with open(fname, 'rb') as f:
        outlist = pickle.load(f)
    return outlist


def save_json_file(outlist, fname):
    with open(fname, 'w') as json_file:
        json.dump(outlist, json_file)


def load_json_file(fname):
    with open(fname, 'r') as json_file:
        file_read = json.load(json_file)
    return file_read


def load_yaml_file(fname):
    with open(fname, 'r') as stream:
        outlist = yaml.safe_load(stream)
    return outlist


def save_yaml_file(outlist, fname):
    with open(fname, 'w') as yaml_file:
        yaml.dump(outlist, yaml_file, default_flow_style=False)


def ragged_tensor_from_nested_numpy(numpy_list):
    """Make ragged tensor from a list of numpy arrays. Ragged dimension only as first axis (ragged_rank=1).

    Args:
        numpy_list (list): List of numpy arrays. Example [np.array, np.array, ...]

    Returns:
        tf.RaggedTensor: Ragged tensor of former nested list of numpy arrays.
    """
    return tf.RaggedTensor.from_row_lengths(np.concatenate(numpy_list, axis=0), np.array([len(x) for x in numpy_list],
                                                                                         dtype=np.int))

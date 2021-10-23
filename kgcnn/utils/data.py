import pickle
import tensorflow as tf
import numpy as np
import yaml
import json
import os
from importlib.machinery import SourceFileLoader


def save_pickle_file(outlist, filepath):
    """Save to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(outlist, f)


def load_pickle_file(filepath):
    """Load pickle file."""
    with open(filepath, 'rb') as f:
        outlist = pickle.load(f)
    return outlist


def save_json_file(outlist, filepath):
    """Save to json file."""
    with open(filepath, 'w') as json_file:
        json.dump(outlist, json_file)


def load_json_file(filepath):
    """Load json file."""
    with open(filepath, 'r') as json_file:
        file_read = json.load(json_file)
    return file_read


def load_yaml_file(fname):
    """Load yaml file."""
    with open(fname, 'r') as stream:
        outlist = yaml.safe_load(stream)
    return outlist


def save_yaml_file(outlist, fname):
    """Save to yaml file."""
    with open(fname, 'w') as yaml_file:
        yaml.dump(outlist, yaml_file, default_flow_style=False)


def load_hyper_file(file_name):
    """Load hyper-parameters from file. File type can be '.yaml', '.json', '.pickle' or '.py'

    Args:
        file_name (str): Path or name of the file containing hyper-parameter.

    Returns:
        hyper (dict): Dictionary of hyper-parameters.
    """
    if "." not in file_name:
        print("ERROR:kgcnn: Can not determine file-type.")
        return {}
    type_ending = file_name.split(".")[-1]
    if type_ending == "json":
        return load_json_file(file_name)
    elif type_ending == "yaml":
        return load_yaml_file(file_name)
    elif type_ending == "pickle":
        return load_pickle_file(file_name)
    elif type_ending == "py":
        path = os.path.realpath(file_name)
        print(path)
        hyper = getattr(SourceFileLoader(os.path.basename(path).replace(".py", ""), path).load_module(), "hyper")
        return hyper
    else:
        print("ERROR:kgcnn: Unsupported file type %s" % type_ending)
    return {}


def ragged_tensor_from_nested_numpy(numpy_list: list):
    """Make ragged tensor from a list of numpy arrays. Each array can have different length but must match in shape
    with exception of the first dimension.
    This will result in a ragged tensor with ragged dimension only at first axis (ragged_rank=1), like
    shape `(batch, None, ...)`. This way a tensor can be generated faster than tf.ragged.constant().

    Warning: The data will be copied for this operation.

    .. code-block:: python

        import tensorflow as tf
        import numpy as np
        ragged_tensor = ragged_tensor_from_nested_numpy([np.array([[0]]), np.array([[1], [2], [3]])])
        print(ragged_tensor)
        # <tf.RaggedTensor [[[0]], [[1], [2], [3]]]>
        print(ragged_tensor.shape)
        # (2, None, 1)

    Args:
        numpy_list (list): List of numpy arrays of different length but else identical shape.

    Returns:
        tf.RaggedTensor: Ragged tensor of former nested list of numpy arrays.
    """
    return tf.RaggedTensor.from_row_lengths(np.concatenate(numpy_list, axis=0), np.array([len(x) for x in numpy_list],
                                                                                         dtype="int"))

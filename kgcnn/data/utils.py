import pickle
import logging
import tensorflow as tf
import numpy as np
import yaml
import json
import os
from importlib.machinery import SourceFileLoader


logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


def save_pickle_file(obj, file_path: str, **kwargs):
    """Save pickle file.

    Args:
        obj: Python-object to dump.
        file_path (str): File path or name to save 'obj' to.

    Returns:
        None.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, **kwargs)


def load_pickle_file(file_path: str, **kwargs):
    """Load pickle file.

    Args:
        file_path (str): File path or name to load.

    Returns:
        obj: Python-object of file.
    """
    with open(file_path, 'rb') as f:
        obj = pickle.load(f, **kwargs)
    return obj


def save_json_file(obj, file_path: str, **kwargs):
    """Save json file.

    Args:
        obj: Python-object to dump.
        file_path (str): File path or name to save 'obj' to.

    Returns:
        None.
    """
    with open(file_path, 'w') as json_file:
        json.dump(obj, json_file, **kwargs)


def load_json_file(file_path: str, **kwargs):
    """Load json file.

    Args:
        file_path (str): File path or name to load.

    Returns:
        obj: Python-object of file.
    """
    with open(file_path, 'r') as json_file:
        file_read = json.load(json_file, **kwargs)
    return file_read


def load_yaml_file(file_path: str):
    """Load yaml file.

    Args:
        file_path (str): File path or name to load.

    Returns:
        obj: Python-object of file.
    """
    with open(file_path, 'r') as stream:
        obj = yaml.safe_load(stream)
    return obj


def save_yaml_file(obj, file_path: str, default_flow_style: bool = False, **kwargs):
    """Save yaml file.

    Args:
        obj: Python-object to dump.
        file_path (str): File path or name to save 'obj' to.
        default_flow_style (bool): Flag for flow style. Default to False.

    Returns:
        None.
    """
    with open(file_path, 'w') as yaml_file:
        yaml.dump(obj, yaml_file, default_flow_style=default_flow_style, **kwargs)


def load_hyper_file(file_name: str, **kwargs) -> dict:
    """Load hyperparameter from file. File type can be '.yaml', '.json', '.pickle' or '.py'.

    Args:
        file_name (str): Path or name of the file containing hyperparameter.

    Returns:
        hyper (dict): Dictionary of hyperparameter.
    """
    if "." not in file_name:
        module_logger.error("Can not determine file-type.")
        return {}
    type_ending = file_name.split(".")[-1]
    if type_ending == "json":
        return load_json_file(file_name, **kwargs)
    elif type_ending == "yaml":
        return load_yaml_file(file_name)
    elif type_ending == "pickle":
        return load_pickle_file(file_name, **kwargs)
    elif type_ending == "py":
        path = os.path.realpath(file_name)
        hyper = getattr(SourceFileLoader(os.path.basename(path).replace(".py", ""), path).load_module(), "hyper")
        return hyper
    else:
        module_logger.error("Unsupported file type '%s'." % type_ending)
    return {}


def ragged_tensor_from_nested_numpy(numpy_list: list, dtype: str = None, row_splits_dtype: str = "int64"):
    r"""Make ragged tensor from a list of numpy arrays. Each array can have different length but must match in shape
    except the first dimension.
    This will result in a ragged tensor with ragged dimension only at first axis (ragged_rank=1), like
    shape `(batch, None, ...)` . This way a tensor can be generated faster than `tf.ragged.constant()` .

    .. warning::
        The data will be copied for this operation.

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
        dtype (str): Data type of values tensor. Defaults to None.
        row_splits_dtype (str): Data type of partition tensor. Default is "int64".

    Returns:
        tf.RaggedTensor: Ragged tensor of former nested list of numpy arrays.
    """
    return tf.RaggedTensor.from_row_lengths(
        np.concatenate(numpy_list, axis=0, dtype=dtype), np.array([len(x) for x in numpy_list], dtype=row_splits_dtype))


def pad_np_array_list_batch_dim(values: list, dtype: str = None):
    r"""Pad a list of numpy arrays along first dimension.

    Args:
        values (list): List of :obj:`np.ndarray` .
        dtype (str): Data type of values tensor. Defaults to None.

    Returns:
        tuple: Padded and mask :obj:`np.ndarray` of values.
    """
    max_shape = np.amax([x.shape for x in values], axis=0)
    final_shape = np.concatenate([np.array([len(values)], dtype="int64"), np.array(max_shape, dtype="int64")])
    padded = np.zeros(final_shape, dtype=values[0].dtype)
    mask = np.zeros(final_shape, dtype="bool")
    for i, x in enumerate(values):
        # noinspection PyTypeChecker
        index = [i] + [slice(0, int(j)) for j in x.shape]
        padded[tuple(index)] = x
        mask[tuple(index)] = True
    if dtype is not None:
        padded = padded.astype(dtype=dtype)
    return padded, mask

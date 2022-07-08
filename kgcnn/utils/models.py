import tensorflow as tf
import functools
import logging
from math import inf
from typing import Union
from copy import deepcopy
import importlib
ks = tf.keras


# Module logger
logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


def get_model_class(module_name: str, class_name: str):
    r"""Helper function to get model class by string identifier.

    Args:
        module_name (str): Name of the module of the model.
        class_name (str): Name of the model class.

    Returns:
        :obj:`tf.keras.models.Model`
    """
    if module_name[:6] != "kgcnn.":
        # Assume that is simply the module name in kgcnn.literature.
        module_name = "kgcnn.literature.%s" % module_name
    if class_name is None or class_name == "":
        # Assume that "make_model" function is used.
        class_name = "make_model"

    try:
        make_model = getattr(importlib.import_module(module_name), class_name)
    except ModuleNotFoundError:
        raise NotImplementedError("Unknown model identifier %s for a model in kgcnn.literature." % class_name)

    return make_model


def generate_embedding(inputs, input_shape: list, embedding_args: dict, embedding_rank: int = 1, **kwargs):
    """Deprecated. Optional embedding for tensor input.
    If there is no feature dimension, an embedding layer can be used.
    If the input tensor has without batch dimension the shape of e.g. `(None, F)` and `F` is the feature dimension,
    no embedding layer is required. However, for shape `(None, )` an embedding with `output_dim` assures a vector
    representation.

    Args:
        inputs (tf.Tensor): Input tensor to make embedding for.
        input_shape (list, tuple): Shape of input without batch dimension. Either (None, F) or (None, ).
        embedding_args (dict): Arguments for embedding layer which will be unpacked in layer constructor.
        embedding_rank (int): The rank of the input which requires embedding. Default is 1.

    Returns:
        tf.Tensor: Tensor embedding dependent on the input shape.
    """
    if len(kwargs) > 0:
        module_logger.warning("Unknown embedding kwargs {0}. Will be reserved for future versions.".format(kwargs))

    if len(input_shape) == embedding_rank:
        n = ks.layers.Embedding(**embedding_args)(inputs)
    else:
        n = inputs
    return n


def update_model_kwargs_logic(default_kwargs: dict = None, user_kwargs: dict = None,
                              update_recursive: Union[int, float] = inf):
    r"""Make model kwargs dictionary with updated default values. This is essentially a nested version of update()
    for dicts. This is supposed to be more convenient if the values of kwargs are again layer kwargs to be unpacked,
    and do not need to be fully known to update them.

    Args:
        default_kwargs (dict): Dictionary of default values. Default is None.
        user_kwargs (dict): Dictionary of args to update. Default is None.
        update_recursive (int): Max depth to update mappings like dict. Default is `inf`.

    Returns:
        dict: New dict and update with first default and then user args.
    """
    if default_kwargs is None:
        default_kwargs = {}
    if user_kwargs is None:
        user_kwargs = {}

    # Check valid kwargs
    for iter_key in user_kwargs.keys():
        if iter_key not in default_kwargs:
            raise ValueError("Model kwarg {0} not in default arguments {1}".format(iter_key, default_kwargs.keys()))

    # Start with default values.
    out = deepcopy(default_kwargs)

    # Nested update of kwargs:
    def _nested_update(dict1, dict2, max_depth=inf, depth=0):
        for key, values in dict2.items():
            if key not in dict1:
                module_logger.warning("Model kwargs: Unknown key {0} with value {1}".format(key, values))
                dict1[key] = values
                continue
            if not isinstance(dict1[key], dict):
                dict1[key] = values
                continue
            if not isinstance(values, dict):
                module_logger.warning("Model kwargs: Overwriting dictionary of {0} with {1}".format(key, values))
                dict1[key] = values
                continue
            # Nested update.
            if depth < max_depth:
                dict1[key] = _nested_update(dict1[key], values, max_depth=max_depth, depth=depth+1)
            else:
                dict1[key] = values
        return dict1

    return _nested_update(out, user_kwargs, update_recursive, 0)


def update_model_kwargs(model_default, update_recursive=inf):
    """Decorating function for update_model_kwargs_logic() ."""
    def model_update_decorator(func):

        @functools.wraps(func)
        def update_wrapper(*args, **kwargs):

            updated_kwargs = update_model_kwargs_logic(model_default, kwargs, update_recursive)

            # Logging of updated values.
            if 'verbose' in updated_kwargs:
                module_logger.setLevel(updated_kwargs["verbose"])
            module_logger.info("Updated model kwargs:")
            module_logger.info(updated_kwargs)

            if len(args) > 0:
                module_logger.error("Can only update kwargs, not %s" % args)

            return func(*args, **updated_kwargs)

        return update_wrapper

    return model_update_decorator

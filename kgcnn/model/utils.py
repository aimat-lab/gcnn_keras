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
        raise NotImplementedError("Unknown model identifier '%s' for a model in kgcnn.literature." % class_name)

    return make_model


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


def change_attributes_in_all_layers(model, attributes_to_change=None, layer_type=None):
    r"""Utility/helper function to change the attributes from a dictionary in all layers of a model of a certain type.

    .. warning::

        This function can change attributes but can cause problems for built models. Also take care which attributes
        you are changing, especially if they include weights. Always check model behaviour after applying this function.

    Args:
        model (tf.keras.models.Model): Model to modify.
        attributes_to_change (dict): Dictionary of attributes to change in all layers of a specific type.
        layer_type: Class type of the layer to change. Default is None.

    Returns:
        tf.keras.models.Model: Model which has layers with changed attributes.
    """
    if model.built:
        module_logger.warning("Model '%s' has already been built. Will set `built=False` and continue." % model.name)
        model.built = False
    if attributes_to_change is None:
        attributes_to_change = {}
    all_layers = model._flatten_layers(include_self=False, recursive=True)
    for x in all_layers:
        if layer_type is not None:
            if not isinstance(x, layer_type):
                continue
        changed_attributes = False
        for key, value in attributes_to_change.items():
            if hasattr(x, key):
                setattr(x, key, value)
                changed_attributes = True
        if changed_attributes:
            if x.built:
                module_logger.warning(
                    "Layer '%s' in model has already been built. Will set `built=False` and continue." % x.name)
                x.built = False
    return model

import importlib
import logging
from typing import Any
# General serialization


def serialize(obj) -> dict:
    """General serialization scheme for objects. Requires module information.
    For each submodule there might be a separate serialization scheme, that works e.g. with name only.

    Args:
        obj: Object to serialize.

    Returns:
        dict: Serialized object.
    """
    obj_dict = dict()
    obj_dict["module_name"] = type(obj).__module__
    obj_dict["class_name"] = type(obj).__name__
    if hasattr(obj, "get_config"):
        obj_dict["config"] = obj.get_config()
    if hasattr(obj, "get_weights"):
        obj_dict["weights"] = obj.get_weights()
    if hasattr(obj, "get_methods"):
        obj_dict["methods"] = obj.get_methods()
    return obj_dict


def deserialize(obj_dict: dict) -> Any:
    r"""General deserialization scheme for objects. Requires module information.
    For each submodule there might be a separate deserialization scheme, that works e.g. with name only.

    Args:
        obj_dict: Serialized object.

    Returns:
        Any: Class or object from :obj:`obj_dict`.
    """

    class_name = obj_dict["class_name"]
    module_name = obj_dict["module_name"]
    try:
        obj_class = getattr(importlib.import_module(str(module_name)), str(class_name))
        config = obj_dict["config"] if "config" in obj_dict else {}
        obj = obj_class(**config)
    except ModuleNotFoundError:
        raise NotImplementedError(
            "Unknown identifier '%s', which is not in modules in kgcnn." % class_name)

    if hasattr(obj, "set_weights") and "weights" in obj_dict:
        obj.set_weights(obj_dict["weights"])

    # Call class methods if methods are in obj_dict.
    # Order is important here.
    if "methods" in obj_dict:
        method_list = obj_dict["methods"]
        if hasattr(obj, "set_methods"):
            obj.set_methods(method_list)
        else:
            # Try setting them manually.
            for method_item in method_list:
                for method, kwargs in method_item.items():
                    if hasattr(obj, method):
                        getattr(obj, method)(**kwargs)
                    else:
                        logging.error("Class for deserialization does not have method '%s'." % method)
    return obj

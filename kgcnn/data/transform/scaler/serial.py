import importlib
from typing import Union
from kgcnn.utils.serial import deserialize as deserialize_general

module_list = {
    "StandardScaler": "kgcnn.data.transform.scaler",
    "StandardLabelScaler": "kgcnn.data.transform.scaler",
    "ExtensiveMolecularScaler": "kgcnn.data.transform.mol",
    "ExtensiveMolecularLabelScaler": "kgcnn.data.transform.mol",
    "EnergyForceExtensiveLabelScaler": "kgcnn.data.transform.force",
    "QMGraphLabelScaler": "kgcnn.data.transform.mol"
}


def deserialize(name: Union[str, dict], **kwargs):
    """Deserialize a scaler class.

    Args:
        name (str, dict): Serialization dictionary of class. This can also be a name of former graph functions for
            backward compatibility that now coincides with the processor's default name.
        kwargs: Kwargs for processor initialization, if :obj:`name` is string.

    Returns:
        GraphPreProcessorBase: Instance of graph preprocessor.
    """

    if isinstance(name, dict):
        if "class_name" not in name:
            raise ValueError("Require 'class_name' for deserialization")
        if "module_name" not in name:
            if name["class_name"] in module_list:
                name["module_name"] = module_list[name["class_name"]]
            else:
                raise ValueError("Unknown module name for serialization for '%s'." % name["class_name"])
        if "config" not in name:
            name["config"] = {}
        return deserialize_general(name)

    if isinstance(name, str):
        # if given as string name. Lookup identifier.
        if name not in module_list:
            raise ValueError("Unknown name for scaler '%s'." % name)
        module_name = module_list[name]
        obj_class = getattr(importlib.import_module(str(module_name)), str(name))
        return obj_class(**kwargs)

    raise TypeError("Wrong type for deserialization. Require 'str' or 'dict'.")
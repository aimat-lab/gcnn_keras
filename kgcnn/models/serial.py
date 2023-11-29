import logging
import keras as ks
from kgcnn.models.utils import get_model_class


def deserialize(obj_dict: dict) -> ks.models.Model:

    if isinstance(obj_dict, ks.models.Model):
        return obj_dict
    if not isinstance(obj_dict, dict):
        raise ValueError("Can not deserialize model with '%s'." % obj_dict)
    if "config" not in obj_dict:
        obj_dict.update({"config": {}})
    class_name = obj_dict.get("class_name")
    module_name = obj_dict.get("module_name")
    model_cls = get_model_class(module_name=module_name, class_name=class_name)
    obj = model_cls(**obj_dict["config"])

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

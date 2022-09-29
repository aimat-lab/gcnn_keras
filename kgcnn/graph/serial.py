import importlib


def get_preprocessor(name, **kwargs):
    preprocessor_identifier = {
        "set_range_periodic": "SetRangePeriodic"
    }
    obj_class = getattr(importlib.import_module(str("kgcnn.graph.preprocessor")), str(preprocessor_identifier[name]))
    return obj_class(**kwargs)
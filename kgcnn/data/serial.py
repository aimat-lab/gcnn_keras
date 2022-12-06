import importlib
import logging
from typing import Union

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

global_dataset_register = {
    "MoleculeNetDataset": {"class_name": "MoleculeNetDataset", "module_name": "kgcnn.data.moleculenet"},
    "QMDataset": {"class_name": "QMDataset", "module_name": "kgcnn.data.qm"},
    "GraphTUDataset": {"class_name": "GraphTUDataset", "module_name": "kgcnn.data.tudataset"},
    "CrystalDataset": {"class_name": "CrystalDataset", "module_name": "kgcnn.data.crystal"}
}


def deserialize(dataset: Union[str, dict]):
    r"""Deserialize a dataset class from dictionary including "class_name" and "config" keys.

    Furthermore, `prepare_data`, `read_in_memory` and `map_list` are possible for deserialization if manually
    set in 'methods' key as list. Tries to resolve datasets also without `module_name` key.
    Otherwise, you can use general `kgcnn.utils.serial` .

    Args:
        dataset (str, dict): Dictionary of the dataset serialization.

    Returns:
        MemoryGraphDataset: Deserialized dataset.
    """
    global global_dataset_register

    # Requires dict. If already deserialized, nothing to do.
    if not isinstance(dataset, (dict, str)):
        module_logger.warning("Can not deserialize dataset %s." % dataset)
        return dataset

    # If only dataset name, make this into a dict with empty config.
    if isinstance(dataset, str):
        dataset = {"class_name": dataset, "config": {}}

    # Find dataset class in register.
    if dataset["class_name"] in global_dataset_register:
        dataset_name = global_dataset_register[dataset["class_name"]]["class_name"]
        module_name = global_dataset_register[dataset["class_name"]]["module_name"]
    else:
        dataset_name = dataset["class_name"]
        module_name = dataset["module_name"] if "module_name" in dataset else "kgcnn.data.datasets.%s" % dataset_name

    try:
        ds_class = getattr(importlib.import_module(str(module_name)), str(dataset_name))
        config = dataset["config"] if "config" in dataset else {}
        ds_instance = ds_class(**config)
    except ModuleNotFoundError:
        raise NotImplementedError(
            "Unknown identifier '%s', which is not in the sub-classed modules in kgcnn.data.datasets" % dataset_name)

    # Call class methods to load or process data.
    # Order is important here.
    if "methods" in dataset:
        method_list = dataset["methods"]
        for method_item in method_list:
            for method, kwargs in method_item.items():
                if hasattr(ds_instance, method):
                    getattr(ds_instance, method)(**kwargs)
                else:
                    ds_instance.error("Dataset class does not have property %s" % method)

    return ds_instance

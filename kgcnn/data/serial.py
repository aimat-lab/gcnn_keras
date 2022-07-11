import importlib
import logging
from typing import Union

try:
    from kgcnn.data.moleculenet import MoleculeNetDataset
except ModuleNotFoundError as e:
    logging.error("Can not import `MoleculeNetDataset` for serialization with %s." % e)
    MoleculeNetDataset = None

try:
    from kgcnn.data.qm import QMDataset
except ModuleNotFoundError as e:
    logging.error("Can not import `QMDataset` for serialization with %s." % e)
    QMDataset = None

try:
    from kgcnn.data.tudataset import GraphTUDataset
except ModuleNotFoundError as e:
    logging.error("Can not import `GraphTUDataset` for serialization with %s." % e)
    GraphTUDataset = None

try:
    from kgcnn.data.crystal import CrystalDataset
except ModuleNotFoundError as e:
    logging.error("Can not import `CrystalDataset` for serialization with %s." % e)
    CrystalDataset = None


logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

global_dataset_register = {
    "MoleculeNetDataset": MoleculeNetDataset,
    "QMDataset": QMDataset,
    "GraphTUDataset": GraphTUDataset,
    "CrystalDataset": CrystalDataset
}


def deserialize(dataset: Union[str, dict]):
    r"""Deserialize a dataset class from dictionary including "class_name" and "config" keys.
    Furthermore, `prepare_data`, `read_in_memory` and `map_list` are possible for deserialization if manually
    set in 'methods' key as list.

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
        ds_class = global_dataset_register[dataset["class_name"]]
        config = dataset["config"] if "config" in dataset else {}
        ds_instance = ds_class(**config)

    # Or load dynamically from datasets folder.
    else:
        dataset_name = dataset["class_name"]
        module_name = dataset["module_name"] if "module_name" in dataset else "kgcnn.data.datasets.%s" % dataset_name
        try:
            ds_class = getattr(importlib.import_module(str(module_name)), str(dataset_name))
            config = dataset["config"] if "config" in dataset else {}
            ds_instance = ds_class(**config)
        except ModuleNotFoundError:
            raise NotImplementedError(
                "Unknown identifier %s, which is not in the sub-classed modules in kgcnn.data.datasets" % dataset_name)

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

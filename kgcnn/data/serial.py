from kgcnn.data.moleculenet import MoleculeNetDataset
from kgcnn.data.qm import QMDataset
from kgcnn.data.tudataset import GraphTUDataset
# from typing import Any
import logging

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

global_dataset_register = {
    "MoleculeNetDataset": MoleculeNetDataset,
    "QMDataset": QMDataset,
    "GraphTUDataset": GraphTUDataset
}

# Add all modules from datasets dynamically here
# ...


def deserialize(dataset: dict):
    r"""Deserialize a dataset class from dictionary including "class_name" and "config" keys.
    Furthermore "prepare_data", "read_in_memory" and "set_attributes" are possible for deserialization.

    Args:
        dataset (dict): Dictionary of the dataset serialization.

    Returns:
        MemoryGraphDataset: Deserialized dataset.
    """
    global global_dataset_register
    # Requires dict. If already deserialized, nothing to do.
    if not isinstance(dataset, dict):
        module_logger.warning("Can not deserialize dataset %s." % dataset)
        return dataset

    # Find dataset class in register.
    ds_class = global_dataset_register[dataset["class_name"]]
    config = dataset["config"] if "config" in dataset else {}
    ds_instance = ds_class(**config)

    # Call class methods to load data.
    method_list = ["prepare_data", "read_in_memory", "set_attributes"]
    for x in method_list:  # Order is important here.
        if x in dataset and hasattr(ds_instance, x):
            getattr(ds_instance, x)(**dataset[x])

    return ds_instance

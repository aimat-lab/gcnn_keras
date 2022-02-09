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
    """Deserialize a dataset class from dictionary including "class_name" and "config" keys. At the moment no loading
    of the data via methods is supported.

    Args:
        dataset (dict): Dictionary of the dataset serialization.

    Returns:
        MemoryGraphDataset: Deserialized dataset.
    """
    global global_dataset_register
    if not isinstance(dataset, dict):
        module_logger.warning("Can not deserialize dataset %s." % dataset)
        return dataset
    ds_class = global_dataset_register[dataset["class_name"]]
    return ds_class(**dataset["config"])

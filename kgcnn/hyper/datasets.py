import os

from copy import deepcopy
from kgcnn.utils.data import load_json_file, load_hyper_file


class DatasetHyperTraining:
    """A class to choose a hyper-parameters for a specific dataset and model. And also to serialize hyper info,
    if possible. Will be added soon.

    """

    def __init__(self, path: str, model_name: str = None, dataset_name: str = None):
        self.dataset_name = dataset_name
        self._hyper_all = load_hyper_file(path)
        self._hyper = None
        if "model" in self._hyper_all and "training" in self._hyper_all:
            self._hyper = self._hyper_all
        elif model_name is not None:
            self._hyper = self._hyper_all[model_name]
        else:
            raise ValueError("ERROR:kgcnn: Not a valid hyper dictionary. Please provide model_name.")

    def get_hyper(self, section = None):
        if section is None:
            return deepcopy(self._hyper)
        else:
            return deepcopy(self._hyper[section])

    def compile(self, loss=None, optimizer=None, metrics = None, weighted_metrics=None):
        pass
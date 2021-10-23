import os

from kgcnn.utils.data import load_json_file, load_hyper_file


class DatasetHyperTraining:
    """A class to choose a hyper-parameters for a specific dataset and model. And also to serialize hyper info,
    if possible. Will be added soon.

    """

    def __init__(self, path):
        self.hyper = load_hyper_file(path)

    def get_hyper(self, model_name: str = None, dataset_name: str = None ):
        if "model" in self.hyper and "training" in self.hyper:
            return self.hyper
        else:
            return self.hyper[model_name]

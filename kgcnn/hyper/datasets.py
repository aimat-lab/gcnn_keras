# import json
import os

from kgcnn.utils.data import load_json_file


class DatasetHyperSelection:

    @classmethod
    def get_hyper(cls, dataset_name: str, model_name: str = None):
        if dataset_name is None:
            return None
        try:
            dir_path = os.path.abspath(os.path.dirname(__file__))
            filepath = os.path.join(dir_path, "hyper_"+dataset_name+".json")
            hyper_dir = load_json_file(filepath)
            if model_name is None:
                return hyper_dir
            if model_name not in hyper_dir:
                raise NotImplementedError("No hyper-parameter for model %s available." % model_name)
            return hyper_dir[model_name]

        except FileNotFoundError:
            raise FileNotFoundError("ERROR:kgcnn: No hyper-parameter for dataset {0} available.".format(dataset_name))

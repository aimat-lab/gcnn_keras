import os
# import json

from kgcnn.utils.data import load_json_file


class DatasetHyperSelection:

    @classmethod
    def get_hyper(cls, dataset_name: str, model_name: str = None):
        if dataset_name is None:
            return None
        try:
            dir_path = os.path.abspath(os.path.dirname(__file__))
            filepath = os.path.join(dir_path, "hyper_" + dataset_name + ".json")
            hyper_dir = load_json_file(filepath)
            if model_name is None:
                return hyper_dir
            if model_name not in hyper_dir:
                raise NotImplementedError("No hyper-parameter for model {0} available, choose models {1} with \
                preset hyper-parameters.".format(model_name, hyper_dir.keys()))
            return hyper_dir[model_name]

        except FileNotFoundError:
            raise FileNotFoundError("ERROR:kgcnn: No hyper-parameter for dataset {0} available.".format(dataset_name))

import json
import os

from kgcnn.utils.data import load_json_file


class DatasetHyperSelection:

    @classmethod
    def get_hyper(cls, dataset_name):
        if dataset_name is None:
            return None
        try:
            dirpath = os.path.abspath(os.path.dirname(__file__))
            filepath = os.path.join(dirpath, "hyper_"+dataset_name+".json")
            hyper_dir = load_json_file(filepath)
            return hyper_dir
        except FileNotFoundError:
            print("ERROR:kgcnn: No hyper-parameter for dataset {0} available.".format(dataset_name))
            return None


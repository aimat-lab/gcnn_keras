import os

from kgcnn.utils.data import load_json_file


class DatasetHyperSelection:
    """A simple class to choose a default set of hyper-parameters for a specific dataset and model.

    Note: Not all datasets and/or models have a matching default hyper-parameter entry yet.
    """

    @classmethod
    def get_hyper(cls, dataset_name: str, model_name: str = None):
        r"""Load the default hyper-parameter from a packaged json-file for a dataset and check for model entry.

        Args:
            dataset_name (str): A dataset that is implemented in :obj:``kgcnn``.
            model_name (str): A model name to get default hyper-parameters for. Default is None.

        Returns:
            dict: Dictionary of hyper-parameters for the requested dataset, if available.
        """
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

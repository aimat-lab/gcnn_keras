import importlib


class DatasetSelection:
    r"""Helper class to load datasets from :obj:`kgcnn.data.datasets`"""

    def __init__(self, dataset_name: str = None):
        r"""Set-up of the :obj:`DatasetSelection` with a name of the dataset to make and modify.

        Args:
            dataset_name (str): Name of the dataset. Default is None.
        """
        super(DatasetSelection, self).__init__()
        self.dataset_name = dataset_name

    def dataset(self, **kwargs):
        r"""Generate a dataset with kwargs for constructor of the dataset. The name of the dataset is passed to this
        class via :obj:`dataset_name`. The actual dataset class if dynamically loaded from a module in
        :obj:`kgcnn.data.datasets`.

        Args:
            kwargs: Kwargs for the dataset constructor.

        Returns:
            MemoryGraphDataset: Sub-classed :obj:`MemoryGraphDataset`.
        """
        dataset_name = self.dataset_name
        if dataset_name is None:
            raise ValueError("A name of the dataset in kgcnn.data.datasets must be provided.")

        try:
            dataset = getattr(importlib.import_module("kgcnn.data.datasets.%s" % dataset_name), str(dataset_name))
            return dataset(**kwargs)
        except ModuleNotFoundError:
            raise NotImplementedError(
                "Unknown identifier %s, which is not in the sub-classed modules in kgcnn.data.datasets" % dataset_name)

    @staticmethod
    def assert_valid_model_input(dataset, hyper_input: list, raise_error_on_fail: bool = True):
        r"""Interface to hyper-parameters. Check whether dataset has graph (tensor) properties requested
        by model input. The model input is set up by a list of layer configs for the keras :obj:`Input` layer.

        Example:
            [{"shape": [None, 8710], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],

        Args:
            dataset: Dataset to validate model input against.
            hyper_input (list): List of properties that need to be available to a model for training.
            raise_error_on_fail (bool): Whether to raise an error if assertion failed.
        """
        def message_error(msg):
            if raise_error_on_fail:
                raise ValueError(msg)
            else:
                dataset.error(msg)

        for x in hyper_input:
            if "name" not in x:
                message_error("Can not infer name from %s for model input." % x)
            data = [dataset[i].obtain_property(x["name"]) for i in range(len(dataset))]
            if any([y is None for y in data]):
                message_error("Property %s is not defined for all graphs in list. Please run clean()." % x["name"])
            # we also will check shape here but only with first element.
            if hasattr(data[0], "shape") and "shape" in x:
                shape_element = data[0].shape
                shape_input = x["shape"]
                if len(shape_input) != len(shape_element):
                    message_error(
                        "Mismatch in rank for model input {} vs. {}".format(shape_element, shape_input))
                for i, dim in enumerate(shape_input):
                    if dim is not None:
                        if shape_element[i] != dim:
                            message_error(
                                "Mismatch in shape for model input {} vs. {}".format(shape_element, shape_input))
            else:
                message_error("Can not check shape for %s." % x["name"])
        return

    @staticmethod
    def perform_methods_on_dataset(dataset, methods_supported: list, hyper_data):
        r"""An interface to hyper-parameters to run further class methods on a :obj:`MemoryGraphDataset`.

        Args:
            dataset: An instance of a :obj:`MemoryGraphDataset`.
            methods_supported (list): List of methods that can be performed on the dataset. The order is respected.
            hyper_data (dict): Dictionary of the 'data' section of hyper-parameters for the dataset.

        Returns:
            MemoryGraphDataset: Modified :obj:`MemoryGraphDataset` from input.
        """
        hyper_data_methods = hyper_data["methods"]

        for method in methods_supported:
            if method in hyper_data_methods:
                if hasattr(dataset, method):
                    getattr(dataset, method)(**hyper_data_methods[method])
                else:
                    # Try if the list is method is on graphs instead on the list.
                    dataset.map_list(method, **hyper_data_methods[method])

        additional_keys = []
        for key, value in hyper_data_methods.items():
            if key not in methods_supported:
                additional_keys.append(key)
        if len(additional_keys) > 0:
            dataset.warning(
                "Additional method(s) found, which do not match a suitable dataset method: %s" % additional_keys)
        return

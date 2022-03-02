from typing import Union
from kgcnn.data.serial import deserialize as deserialize_dataset


class DatasetSelection:
    r"""Helper class to load datasets from :obj:`kgcnn.data.datasets`. Note that datasets can be deserialized via
    `kgcnn.data.serial.deserialize`. This class is used for accessing datasets in training scripts.

    """

    def __init__(self, dataset_name: str = None):
        r"""Set-up of :obj:`DatasetSelection` with a name of the dataset.

        Args:
            dataset_name (str): Name of the dataset. Default is None.
        """
        super(DatasetSelection, self).__init__()
        self.dataset_name = dataset_name

    def dataset(self, **kwargs):
        r"""Generate a dataset with kwargs for construction of the dataset. The name of the dataset is stored in
        class property :obj:`dataset_name`. The actual dataset class is dynamically loaded from a module in
        :obj:`kgcnn.data.datasets`.

        Args:
            kwargs: Kwargs for the dataset config. This can be either directly the kwargs of "config" or the dictionary
                of {"config": {...}, "methods": [...], ...} itself unpacked, which is preferred.

        Returns:
            MemoryGraphDataset: Sub-classed :obj:`MemoryGraphDataset`.
        """
        if self.dataset_name is None:
            raise ValueError("A name of the dataset in kgcnn.data.datasets must be provided.")
        dataset_config = {"class_name": self.dataset_name}
        # kwargs can be directly config of dataset or within the parameter config.
        if "config" in kwargs:
            if "class_name" in kwargs:
                if kwargs["class_name"] != self.dataset_name:
                    raise ValueError("Specified dataset does not match with kwargs %s" % self.dataset_name)
            dataset_config.update(kwargs)
        else:
            dataset_config.update({"config": kwargs})
        return deserialize_dataset(dataset_config)

    @staticmethod
    def assert_valid_model_input(dataset, hyper_input: list, raise_error_on_fail: bool = True):
        r"""Interface to hyperparameter. Check whether dataset has graph-properties (tensor format) requested
        by model input. The model input is set up by a list of layer configs for the keras :obj:`Input` layer.

        Example:
            [{"shape": [None, 8710], "name": "node_attributes", "dtype": "float32", "ragged": True},
            {"shape": [None, 1], "name": "edge_weights", "dtype": "float32", "ragged": True},
            {"shape": [None, 2], "name": "edge_indices", "dtype": "int64", "ragged": True}],

        Args:
            dataset (MemoryGraphDataset): Dataset to validate model input against.
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

    # Deprecated. Keep for compatibility.
    @staticmethod
    def perform_methods_on_dataset(dataset, methods_supported: list, hyper_data: dict,
                                   ignore_unsupported: bool = True):
        r"""An interface to hyperparameter to run further methods with :obj:`MemoryGraphDataset`.
        This method is deprecated and will not be used for further development in favor of
        `kgcnn.data.serial.deserialize`.

        Args:
            dataset (MemoryGraphDataset): An instance of a :obj:`MemoryGraphDataset`.
            methods_supported (list): List of methods that can be performed on the dataset. The order is respected.
            hyper_data (dict): Dictionary of the 'data' section of hyperparameter for the dataset. Must contain an
                item "methods".
            ignore_unsupported (bool): Ignore methods that are not in `methods_supported`.

        Returns:
            MemoryGraphDataset: Modified :obj:`MemoryGraphDataset` from input.
        """
        hyper_data_methods = hyper_data["methods"] if "methods" in hyper_data else {}
        unsupported_methods = []

        methods_to_run = []
        # Case where "methods": {"method": {...}, ...}
        if isinstance(hyper_data_methods, dict):
            # First append supported methods in order of methods_supported.
            for method_item in methods_supported:
                if method_item in hyper_data_methods:
                    methods_to_run.append({method_item: hyper_data_methods[method_item]})
            # Append methods that are not in methods_supported afterwards.
            for method, kwargs in hyper_data_methods.items():
                if method not in methods_supported:
                    methods_to_run.append({method: kwargs})
        # Case where "methods": [{"method": {...}}, {...}, ...]
        elif isinstance(hyper_data_methods, list):
            methods_to_run = hyper_data_methods
        else:
            dataset.error("Methods defined in hyper must be list or dict but got %s" % type(hyper_data_methods))
            TypeError("Wrong type of method list for dataset.")

        # Try to call all methods in methods_to_run.
        for method_item in methods_to_run:
            for method, kwargs in method_item.items():
                if method not in methods_supported:
                    unsupported_methods.append(method)
                    if ignore_unsupported:
                        continue
                # Check if dataset has method.
                if hasattr(dataset, method):
                    getattr(dataset, method)(**kwargs)
                # Else pass along to each graph in list via map_list.
                else:
                    dataset.map_list(method, **kwargs)

        if len(unsupported_methods) > 0:
            dataset.warning(
                "Additional method(s) found, which do not match a suitable dataset method: %s" % unsupported_methods)
        return

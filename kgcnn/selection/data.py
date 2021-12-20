import importlib

class DatasetSelection:

    def __init__(self, dataset_name: str = None):
        super(DatasetSelection, self).__init__()
        self.dataset_name = dataset_name

    def dataset(self, **kwargs):
        dataset_name = self.dataset_name

        try:
            dataset = getattr(importlib.import_module("kgcnn.data.datasets.%s" % dataset_name), str(dataset_name))
            return dataset(**kwargs)
        except ModuleNotFoundError:
            raise NotImplementedError("ERROR:kgcnn: Unknown dataset identifier %s" % dataset_name)

    @staticmethod
    def assert_valid_model_input(dataset, hyper_input: list, raise_error_on_fail: bool = True):
        """Interface to hyper-parameters. Check whether dataset has requested graph (tensor) properties requested
        by model input.

        Args:
            dataset: Dataset to chek.
            hyper_input (list): List of properties that need to be available to a model for training.
            raise_error_on_fail (bool): Whether to raise an error if assertion failed.
        """
        def message_error(msg):
            if raise_error_on_fail:
                raise ValueError(msg)
            else:
                print("ERROR:", msg)

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
    def perform_methods_on_dataset(dataset, methods_supported, hyper_data):
        for method in methods_supported:
            if method in hyper_data:
                if hasattr(dataset, method):
                    getattr(dataset, method)(**hyper_data[method])
                else:
                    dataset.map_list(method, **hyper_data[method])

        additional_keys = []
        for key, value in hyper_data.items():
            if key not in methods_supported:
                additional_keys.append(key)
        if len(additional_keys) > 0:
            dataset.warning(
                "Additional key(s) found, which does not match a suitable dataset method: %s" % additional_keys)
        return

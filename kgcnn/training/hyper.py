import os
import logging
import keras as ks
from typing import Union
from copy import deepcopy
from kgcnn.metrics.utils import merge_metrics
from kgcnn.data.utils import load_hyper_file, save_json_file
from keras.saving import deserialize_keras_object
import keras.metrics
import keras.losses

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


class HyperParameter:
    r"""A class to store hyperparameter for a specific dataset and model, exposing them for model training scripts.

    This includes training parameters and a set of general information like a path for output of the training stats
    or the expected version of :obj:`kgcnn`. The class methods will extract and possibly serialize or deserialize the
    necessary kwargs from the hyperparameter dictionary.

    .. code-block:: python

        hyper = HyperParameter(hyper_info={"model": {"config":{}}, "training": {}, "data":{"dataset":{}}})

    """

    def __init__(self, hyper_info: Union[str, dict],
                 hyper_category: str = None,
                 model_name: str = None,
                 model_module: str = None,
                 model_class: str = "make_model",
                 dataset_name: str = None,
                 dataset_class: str = None,
                 dataset_module: str = None):
        r"""Make a hyperparameter class instance. Required is the hyperparameter dictionary or path to a config
        file containing the hyperparameter. Furthermore, name of the dataset and model can be provided for checking
        the information in :obj:`hyper_info`.

        Args:
            hyper_info (str, dict): Hyperparameter dictionary or path to file.
            hyper_category (str): Category within hyperparameter.
            model_name (str, optional): Name of the model.
            model_module (str, optional): Name of the module of the model. Defaults to None.
            model_class (str, optional): Class name or make function for model. Defaults to 'make_model'.
            dataset_name (str, optional): Name of the dataset.
            dataset_class (str, optional): Class name of the dataset.
            dataset_module (str, optional): Module name of the dataset.
        """
        self._hyper = None
        self._hyper_category = hyper_category
        self._dataset_name = dataset_name
        self._model_name = model_name
        self._model_module = model_module
        self._model_class = model_class
        self._dataset_name = dataset_name
        self._dataset_class = dataset_class
        self._dataset_module = dataset_module

        if isinstance(hyper_info, str):
            self._hyper_all = load_hyper_file(hyper_info)
        elif isinstance(hyper_info, dict):
            self._hyper_all = hyper_info
        else:
            raise TypeError("`HyperParameter` requires valid hyper dictionary or path to file.")

        # If model and training section in hyper-dictionary, then this is a valid hyper setting.
        # If hyper is itself a dictionary with many models, pick the right model if model name is given.
        model_name_class = "%s.%s" % (model_name, model_class)
        if "model" in self._hyper_all and "training" in self._hyper_all:
            self._hyper = self._hyper_all
        elif hyper_category is not None:
            if hyper_category in self._hyper_all:
                self._hyper = self._hyper_all[self._hyper_category]
            else:
                raise ValueError("Category '%s' not in hyperparameter information." % hyper_category)
        elif model_name is not None and model_class == "make_model" and model_name in self._hyper_all:
            self._hyper = self._hyper_all[model_name]
        elif model_name is not None and model_class != "make_model" and model_name_class in self._hyper_all:
            self._hyper = self._hyper_all[model_name_class]
        else:
            raise ValueError("".join(
                ["Not a valid hyper dictionary. If there are multiple hyperparameter settings in `hyper_info`, ",
                 "please specify `hyper_category` or the category is inferred from optional class/name ",
                 "information but which is deprecated."]))

    def verify(self, raise_error: bool = True, raise_warning: bool = False):
        """Logic to verify and optionally update hyperparameter dictionary."""

        def error_msg(error_to_report):
            if raise_error:
                raise ValueError(error_to_report)
            else:
                module_logger.error(error_to_report)

        def warning_msg(warning_to_report):
            if raise_warning:
                raise ValueError(warning_to_report)
            else:
                module_logger.warning(warning_to_report)

        # Update some optional parameters in hyper. Issue a warning.
        if "config" not in self._hyper["model"] and "inputs" in self._hyper["model"]:
            # Older model config setting.
            warning_msg("Hyperparameter {'model': ...} changed to {'model': {'config': {...}}} .")
            self._hyper["model"] = {"config": deepcopy(self._hyper["model"])}
        if "class_name" not in self._hyper["model"] and self._model_class is not None:
            warning_msg("Adding model class from self to 'model': {'class_name': %s} ." % self._model_class)
            self._hyper["model"].update({"class_name": self._model_class})
        if "module_name" not in self._hyper["model"] and self._model_module is not None:
            warning_msg("Adding model module from self to 'model': {'module_name': %s} ." % self._model_module)
            self._hyper["model"].update({"module_name": self._model_module})
        if "info" not in self._hyper:
            self._hyper.update({"info": {}})
            warning_msg("Adding 'info' category to hyperparameter.")
        if "postfix_file" not in self._hyper["info"]:
            warning_msg("Adding 'postfix_file' to 'info' category in hyperparameter.")
            self._hyper["info"].update({"postfix_file": ""})
        if "dataset" in self._hyper["data"] and "dataset" not in self._hyper:
            warning_msg("Hyperparameter should have separate dataset category from kgcnn>=3.1.0 .")
            self._hyper["dataset"] = deepcopy(self._hyper["data"]["dataset"])
            self._hyper["data"].pop("dataset")

        # Errors if missmatch is found between class definition and information in hyper-dictionary.
        # In principle all information regarding model and dataset can be stored in hyper-dictionary.
        if "class_name" in self._hyper["dataset"] and self._dataset_class is not None:
            if self._dataset_class != self._hyper["dataset"]["class_name"]:
                error_msg("Dataset '%s' does not agree with hyperparameter '%s'." % (
                    self._dataset_class, self._hyper["dataset"]["class_name"]))
        if "class_name" in self._hyper["model"] and self._model_class is not None:
            if self._hyper["model"]["class_name"] != self._model_class:
                error_msg("Model generation '%s' does not agree with hyperparameter '%s'." % (
                    self._model_class, self._hyper["model"]["class_name"]))
        if "module_name" in self._hyper["model"] and self._model_module is not None:
            if self._hyper["model"]["module_name"] != self._model_module:
                error_msg("Model module '%s' does not agree with hyperparameter '%s'." % (
                    self._model_module, self._hyper["model"]["module_name"]))
        if "name" in self._hyper["model"]["config"] and self._model_name is not None:
            if self._hyper["model"]["config"]["name"] != self._model_name:
                error_msg("Model name '%s' does not agree with hyperparameter '%s'." % (
                    self._model_name, self._hyper["model"]["config"]["name"]))

    def __getitem__(self, item):
        return deepcopy(self._hyper[item])

    def compile(self, loss=None, optimizer='rmsprop', metrics: list = None, weighted_metrics: list = None):
        r"""Generate kwargs for :obj:`tf.keras.Model.compile` from hyperparameter and default parameter.

        This function should handle deserialization of hyperparameter and, if not specified, fill them from default.
        Loss, optimizer are overwritten from hyperparameter, if available. Metrics in hyperparameter are added from
        function arguments. Note that otherwise metrics can not be deserialized, since `metrics` can include nested
        lists and a dictionary of model output names.

        .. warning::

            When using deserialization with this function, you must not name your model output "class_name",
            "module_name" and "config".

        Args:
            loss: Default loss for fit. Default is None.
            optimizer: Default optimizer. Default is "rmsprop".
            metrics (list): Default list of metrics. Default is None.
            weighted_metrics (list): Default list of weighted_metrics. Default is None.

        Returns:
            dict: Deserialized compile kwargs from hyperparameter.
        """
        hyper_compile = deepcopy(self._hyper["training"]["compile"]) if "compile" in self._hyper["training"] else {}
        if len(hyper_compile) == 0:
            module_logger.warning("Found no information for `compile` in hyperparameter.")
        reserved_compile_arguments = ["loss", "optimizer", "weighted_metrics", "metrics"]
        hyper_compile_additional = {
            key: value for key, value in hyper_compile.items() if key not in reserved_compile_arguments}

        def nested_deserialize(m, get):
            """Deserialize nested list or dict objects for keras model output like loss or metrics."""
            if isinstance(m, (list, tuple)):
                return [nested_deserialize(x, get) for x in m]
            if isinstance(m, dict):
                if "class_name" in m and "config" in m:  # Here we have a serialization dict.
                    try:
                        return get(m)
                    except ValueError:
                        return deserialize_keras_object(m)
                else:  # Model outputs as dict.
                    return {key: nested_deserialize(value, get) for key, value in m.items()}
            elif isinstance(m, str):
                return get(m)
            return m

        # Optimizer
        optimizer = nested_deserialize(
            (hyper_compile["optimizer"] if "optimizer" in hyper_compile else optimizer), ks.optimizers.get)

        # Loss
        loss = nested_deserialize((hyper_compile["loss"] if "loss" in hyper_compile else loss), ks.losses.get)

        # Metrics
        metrics = nested_deserialize(metrics, ks.metrics.get)
        weighted_metrics = nested_deserialize(weighted_metrics, ks.metrics.get)
        hyper_metrics = nested_deserialize(
            hyper_compile["metrics"], ks.metrics.get) if "metrics" in hyper_compile else None
        hyper_weighted_metrics = nested_deserialize(
            hyper_compile["weighted_metrics"], ks.metrics.get) if "weighted_metrics" in hyper_compile else None

        metrics = merge_metrics(hyper_metrics, metrics)
        weighted_metrics = merge_metrics(hyper_weighted_metrics, weighted_metrics)

        # Output deserialized compile kwargs.
        output = {"loss": loss, "optimizer": optimizer, "metrics": metrics, "weighted_metrics": weighted_metrics,
                  **hyper_compile_additional}
        module_logger.info("Deserialized compile kwargs '%s'." % output)
        return output

    def fit(self, epochs: int = 1, validation_freq: int = 1, batch_size: int = None, callbacks: list = None):
        """Select fit hyperparameter. Additional default values for the training scripts are given as
        functional kwargs. Functional kwargs are overwritten by hyperparameter.

        Args:
            epochs (int): Default number of epochs. Default is 1.
            validation_freq (int): Default validation frequency. Default is 1.
            batch_size (int): Default batch size. Default is None.
            callbacks (list): Default Callbacks. Default is None.

        Returns:
            dict: de-serialized fit kwargs from hyperparameter.
        """
        hyper_fit = deepcopy(self._hyper["training"]["fit"])

        reserved_fit_arguments = ["callbacks", "batch_size", "validation_freq", "epochs"]
        hyper_fit_additional = {key: value for key, value in hyper_fit.items() if key not in reserved_fit_arguments}

        if "epochs" in hyper_fit:
            epochs = hyper_fit["epochs"]
        if "validation_freq" in hyper_fit:
            validation_freq = hyper_fit["validation_freq"]
        if "batch_size" in hyper_fit:
            batch_size = hyper_fit["batch_size"]
        if "callbacks" in hyper_fit:
            if callbacks is None:
                callbacks = []
            for cb in hyper_fit["callbacks"]:
                if isinstance(cb, (str, dict)):
                    callbacks += [deserialize_keras_object(cb)]
                else:
                    callbacks += [cb]

        out = {"batch_size": batch_size, "epochs": epochs, "validation_freq": validation_freq, "callbacks": callbacks}
        out.update(hyper_fit_additional)
        return out

    def results_file_path(self):
        r"""Make output folder for results based on hyperparameter and return path to that folder.
        The folder is set up as `'results'/dataset/model_name + post_fix`. Where model and dataset name must be set by
        this class. Postfix can be in hyperparameter setting.

        Returns:
            str: File-path or path object to result folder.
        """
        hyper_info = deepcopy(self._hyper["info"])
        post_fix = str(hyper_info["postfix"]) if "postfix" in hyper_info else ""
        os.makedirs("results", exist_ok=True)
        os.makedirs(os.path.join("results", self.dataset_class), exist_ok=True)
        category = self.hyper_category.replace(".", "_")
        filepath = os.path.join("results", self.dataset_class, category + post_fix)
        os.makedirs(filepath, exist_ok=True)
        return filepath

    def save(self, file_path: str):
        """Save the hyperparameter to path.

        Args:
            file_path (str): Full file path to save hyperparameter to.
        """
        # Must make more refined saving and serialization here.
        save_json_file(self._hyper, file_path)

    @property
    def dataset_name(self):
        if "dataset" in self._hyper:
            if "config" in self._hyper["dataset"]:
                if "dataset_name" in self._hyper["dataset"]["config"]:
                    return self._hyper["dataset"]["config"]["dataset"]
        return self._dataset_name

    @property
    def dataset_class(self):
        if "dataset" in self._hyper:
            if "class_name" in self._hyper["dataset"]:
                return self._hyper["dataset"]["class_name"]
        return self._dataset_name

    @property
    def dataset_module(self):
        if "dataset" in self._hyper:
            if "module_name" in self._hyper["dataset"]:
                return self._hyper["dataset"]["module_name"]
        return self._dataset_name

    @property
    def model_name(self):
        if "model" in self._hyper:
            if "config" in self._hyper["model"]:
                if "name" in self._hyper["model"]["config"]:
                    # Name is just called name not model_name to be more compatible with keras.
                    return self._hyper["model"]["config"]["name"]
        return self._model_name

    @property
    def model_class(self):
        if "model" in self._hyper:
            if "class_name" in self._hyper["model"]:
                return self._hyper["model"]["class_name"]
        return self._model_class

    @property
    def model_module(self):
        if "model" in self._hyper:
            if "module_name" in self._hyper["model"]:
                return self._hyper["model"]["module_name"]
        return self._model_module

    @property
    def hyper_category(self):
        if self._hyper_category is not None:
            return self._hyper_category
        return "%s.%s" % (self.model_name, self.model_class)

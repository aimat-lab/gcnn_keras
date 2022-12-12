import tensorflow as tf
import os
import logging
from typing import Union
from copy import deepcopy
from kgcnn.data.utils import load_hyper_file, save_json_file

ks = tf.keras

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
                 model_name: str = None, model_module: str = None,
                 model_class: str = "make_model", dataset_name: str = None, dataset_class: str = None,
                 dataset_module: str = None):
        r"""Make a hyperparameter class instance. Required is the hyperparameter dictionary or path to a config
        file containing the hyperparameter. Furthermore, name of the dataset and model can be provided for checking
        the information in :obj:`hyper_info`.

        Args:
            hyper_info (str, dict): Hyperparameter dictionary or path to file.
            model_name (str): Name of the model.
            model_module (str): Name of the module of the model. Defaults to None.
            model_class (str): Class name or make function for model. Defaults to 'make_model'.
            dataset_name (str): Name of the dataset.
            dataset_class (str): Class name of the dataset.
            dataset_module (str): Module name of the dataset.
        """
        self._hyper = None
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model_module = model_module
        self.model_class = model_class
        self.dataset_name = dataset_name
        self.dataset_class = dataset_class
        self.dataset_module = dataset_module

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
        elif model_name is not None and model_class == "make_model" and model_name in self._hyper_all:
            self._hyper = self._hyper_all[model_name]
        elif model_name is not None and model_class != "make_model" and model_name_class in self._hyper_all:
            self._hyper = self._hyper_all[model_name_class]
        else:
            raise ValueError("".join(
                ["Not a valid hyper dictionary. If there are multiple hyperparameter settings in `hyper_info`,",
                 "please set `model_name` and `model_class`."]))

        self.verify()

    def verify(self):
        """Logic to verify and optionally update hyperparameter dictionary."""

        # Update some optional parameters in hyper.
        if "config" not in self._hyper["model"] and "inputs" in self._hyper["model"]:
            # Older model config setting.
            module_logger.warning("Hyperparameter {'model': ...} changed to {'model': {'config': {...}}}")
            self._hyper["model"] = {"config": deepcopy(self._hyper["model"])}
        if "class_name" not in self._hyper["model"] and self.model_class is not None:
            module_logger.info("Adding model class from self to 'model': {'class_name': %s}" % self.model_class)
            self._hyper["model"].update({"class_name": self.model_class})
        if "module_name" not in self._hyper["model"] and self.model_module is not None:
            module_logger.info("Adding model module from self to 'model': {'module_name': %s}" % self.model_module)
            self._hyper["model"].update({"module_name": self.model_module})
        if "info" not in self._hyper:
            self._hyper.update({"info": {}})
            module_logger.info("Adding 'info' category to hyperparameter.")
        if "postfix_file" not in self._hyper["info"]:
            module_logger.info("Adding 'postfix_file' to 'info' category in hyperparameter.")
            self._hyper["info"].update({"postfix_file": ""})

        # Errors if missmatch is found between class definition and information in hyper-dictionary.
        # In principle all information regarding model and dataset can be stored in hyper-dictionary.
        if "class_name" in self._hyper["data"]["dataset"] and self.dataset_name is not None:
            if self.dataset_name != self._hyper["data"]["dataset"]["class_name"]:
                raise ValueError(
                    "Dataset '%s' does not agree with hyperparameter '%s'" % (
                        self.dataset_name, self._hyper["data"]["dataset"]["class_name"]))
        if "class_name" in self._hyper["model"] and self.model_class is not None:
            if self._hyper["model"]["class_name"] != self.model_class:
                raise ValueError(
                    "Model generation '%s' does not agree with hyperparameter '%s'" % (
                        self.model_class, self._hyper["model"]["class_name"]))
        if "module_name" in self._hyper["model"] and self.model_module is not None:
            if self._hyper["model"]["module_name"] != self.model_module:
                raise ValueError(
                    "Model module '%s' does not agree with hyperparameter '%s'" % (
                        self.model_module, self._hyper["model"]["module_name"]))
        if "name" in self._hyper["model"]["config"] and self.model_name is not None:
            if self._hyper["model"]["config"]["name"] != self.model_name:
                module_logger.error(
                    "Model name '%s' does not agree with hyperparameter '%s'" % (
                        self.model_name, self._hyper["model"]["config"]["name"]))

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
                        return ks.utils.deserialize_keras_object(m)
                else:  # Model outputs as dict.
                    return {key: nested_deserialize(value, get) for key, value in m.items()}
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

        def merge_metrics(m1, m2):
            """Merge two metric lists or dicts of `ks.metrics` objects."""
            if m1 is None:
                return m2
            if m2 is None:
                return m1
            # Dict case with multiple named outputs.
            if isinstance(m1, dict) and isinstance(m2, dict):
                keys = set(list(m1.keys()) + list(m2.keys()))
                m = {key: [] for key in keys}
                for mu in [m1, m2]:
                    for key, value in mu.items():
                        if value is not None:
                            m[key] = m[key] + (list(value) if isinstance(value, (list, tuple)) else [value])
                return m
            # Lists for single model output.
            m1 = [m1] if not isinstance(m1, (list, tuple)) else m1
            m2 = [m2] if not isinstance(m2, (list, tuple)) else m2
            if all([not isinstance(x1, (list, tuple)) for x1 in m1] + [not isinstance(x2, (list, tuple)) for x2 in m2]):
                return m1 + m2
            # List for multiple model output with nested lists.
            if len(m1) == len(m2):
                m = [[]] * len(m1)
                for i in range(len(m)):
                    for mu in [m1, m2]:
                        if mu[i] is not None:
                            m[i] = m[i] + (list(mu[i]) if isinstance(mu[i], (list, tuple)) else [mu[i]])
                return m
            else:
                module_logger.error("For multiple model outputs require same length of metrics list to merge.")
            module_logger.error("Can not merge metrics '%s' and '%s'." % (m1, m2))
            return None

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
                try:
                    callbacks += [tf.keras.utils.deserialize_keras_object(cb)]
                except:
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
        os.makedirs(os.path.join("results", self.dataset_name), exist_ok=True)
        model_name = self._hyper['model']['config']['name']
        if "class_name" in self._hyper['model']:
            class_name = self._hyper['model']["class_name"]
            if class_name != "make_model":
                model_name = "%s_%s" % (model_name, class_name)
        filepath = os.path.join("results", self.dataset_name, model_name + post_fix)
        os.makedirs(filepath, exist_ok=True)
        return filepath

    def save(self, file_path: str):
        """Save the hyperparameter to path.

        Args:
            file_path (str): Full file path to save hyperparameter to.
        """
        # Must make more refined saving and serialization here.
        save_json_file(self._hyper, file_path)

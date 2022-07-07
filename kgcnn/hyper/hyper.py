import tensorflow as tf
import os
import logging
from typing import Union
from copy import deepcopy
from kgcnn.data.utils import load_hyper_file, save_json_file

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


class HyperParameter:
    r"""A class to store hyperparameter for a specific dataset and model, exposing them for model training scripts.
    This includes training parameters and a set of general information like a path for output of the training stats
    or the expected version of :obj:`kgcnn`. The class methods will extract and possibly serialize or deserialized the
    necessary kwargs from the hyperparameter dictionary.

    """

    def __init__(self, hyper_info: Union[str, dict],
                 model_name: str = None, dataset_name: str = None, model_generation: str = None):
        """Make a hyperparameter class instance. Required is the hyperparameter dictionary or path to a config
        file containing the hyperparameter. Furthermore, name of the dataset and model can be provided.

        Args:
            hyper_info (str, dict): Hyperparameter dictionary or path to file.
            model_name (str): Name or module of the model.
            model_generation (str): Class name or make function for model.
            dataset_name (str): Name of the dataset.
        """
        self._hyper = None
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model_module = model_name
        self.model_generation = model_generation

        if isinstance(hyper_info, str):
            self._hyper_all = load_hyper_file(hyper_info)
        elif isinstance(hyper_info, dict):
            self._hyper_all = hyper_info
        else:
            raise TypeError("`HyperParameter` requires valid hyper dictionary or path to file.")

        # If model and training section in hyper-dictionary, then this is a valid hyper setting.
        # If hyper is itself a dictionary with many models, pick the right model if model name is given.
        if "model" in self._hyper_all and "training" in self._hyper_all:
            self._hyper = self._hyper_all
        elif model_name is not None and model_name in self._hyper_all:
            self._hyper = self._hyper_all[model_name]
        else:
            raise ValueError("Not a valid hyper dictionary. Please provide model_name.")

        # Check hyperparameter
        if "config" not in self._hyper["model"] and "inputs" in self._hyper["model"]:
            module_logger.warning("Hyperparameter {'model': ...} changed to {'model': {'config': {...}}}")
            self._hyper["model"] = {"config": deepcopy(self._hyper["model"])}
        if "class_name" not in self._hyper["model"] and self.model_generation is not None:
            module_logger.info("Adding model class to 'model': {'class_name': %s}" % self.model_generation)
            self._hyper["model"].update({"class_name": self.model_generation})
        if "info" not in self._hyper:
            self._hyper.update({"info": {}})
            module_logger.info("Adding 'info' category to hyperparameter.")
        if "postfix_file" not in self._hyper["info"]:
            module_logger.info("Adding 'postfix_file' to 'info' category in hyperparameter.")
            self._hyper["info"].update({"postfix_file": ""})
        if "multi_target_indices" not in self._hyper["training"]:
            module_logger.info("Adding 'multi_target_indices' to 'training' category in hyperparameter.")
            self._hyper["training"].update({"multi_target_indices": None})
        if "data_unit" not in self._hyper["data"]:
            module_logger.info("Adding 'data_unit' to 'data' category in hyperparameter.")
            self._hyper["data"].update({"data_unit": ""})

    def __getitem__(self, item):
        return deepcopy(self._hyper[item])

    def compile(self, loss=None, optimizer='rmsprop', metrics: list = None, weighted_metrics: list = None):
        """Select compile hyperparameter. Additional default values for the training scripts are given as
        functional kwargs. Functional kwargs are overwritten by hyperparameter.

        Args:
            loss: Default loss for fit. Default is None.
            optimizer: Default optimizer. Default is "rmsprop".
            metrics (list): Default list of metrics. Default is None.
            weighted_metrics (list): Default list of weighted_metrics. Default is None.

        Returns:
            dict: Deserialized compile kwargs from hyperparameter.
        """
        hyper_compile = deepcopy(self._hyper["training"]["compile"])
        reserved_compile_arguments = ["loss", "optimizer", "weighted_metrics", "metrics"]
        hyper_compile_additional = {key: value for key, value in hyper_compile.items() if
                                    key not in reserved_compile_arguments}
        if "optimizer" in hyper_compile:
            try:
                optimizer = tf.keras.optimizers.get(hyper_compile['optimizer'])
            except:
                optimizer = hyper_compile['optimizer']
        if "loss" in hyper_compile:
            loss = hyper_compile["loss"]
        if "weighted_metrics" in hyper_compile:
            if weighted_metrics is None:
                weighted_metrics = []
            weighted_metrics += [x for x in hyper_compile["weighted_metrics"]]
        if "metrics" in hyper_compile:
            if metrics is None:
                metrics = []
            metrics += [x for x in hyper_compile["metrics"]]

        out = {"loss": loss, "optimizer": optimizer, "metrics": metrics, "weighted_metrics": weighted_metrics}
        out.update(hyper_compile_additional)
        return out

    def fit(self, epochs: int = 1, validation_freq: int = 1, batch_size: int = None, callbacks: list = None):
        """Select fit hyperparameter. Additional default values for the training scripts are given as
        functional kwargs. Functional kwargs are overwritten by hyperparameters.

        Args:
            epochs (int): Default number of epochs. Default is 1.
            validation_freq (int): Default validation frequency. Default is 1.
            batch_size (int): Default batch size. Default is None.
            callbacks (list): Default Callbacks. Default is None.

        Returns:
            dict: de-serialized fit kwargs from hyperparameters.
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

import tensorflow as tf
import os
import numpy as np

from copy import deepcopy
from kgcnn.utils.data import load_hyper_file, save_json_file


class HyperSelection:
    r"""A class to store hyper-parameters for a specific dataset and model, exposing them for model training scripts.
    This includes training parameters and a set of general information like a path for output of the training stats
    or the expected version of `kgcnn`. The class methods will extract and possibly serialize or deserialized the
    necessary kwargs from the hyper-parameter dictionary with additional default values. Also changes in the
    hyper-parameter definition can be made without affecting training scripts and made compatible with previous
    versions.
    """

    def __init__(self, hyper_info: str, model_name: str = None, dataset_name: str = None):
        """Make a hyper-parameter class instance. Required is the hyper-parameter dictionary or path to a config
        file containing the hyper-parameters. Furthermore name of the dataset and model can be provided.

        Args:
            hyper_info (str, dict): Hyper-parameters dictionary or path to file.
            model_name (str): Name of the model.
            dataset_name (str): Name of the dataset.
        """

        self.dataset_name = dataset_name
        self.model_name = model_name

        if isinstance(hyper_info, str):
            self._hyper_all = load_hyper_file(hyper_info)
        elif isinstance(hyper_info, dict):
            self._hyper_all = hyper_info
        else:
            raise TypeError("`HyperSelection` requires valid hyper dictionary or path to file.")

        self._hyper = None
        # If model and training section in hyper-dictionary, then this is a valid hyper setting.
        # If hyper is itself a dictionary with many models, pick the right model if model name is given.
        if "model" in self._hyper_all and "training" in self._hyper_all:
            self._hyper = self._hyper_all
        elif model_name is not None and model_name in self._hyper_all:
            self._hyper = self._hyper_all[model_name]
        elif dataset_name is not None and dataset_name in self._hyper_all:
            self._hyper = self._hyper_all[dataset_name]
        else:
            raise ValueError("Not a valid hyper dictionary. Please provide model_name.")

    def hyper(self):
        """Return copy of all hyper-parameters"""
        return deepcopy(self._hyper)

    def model(self):
        """Return copy of model section of hyper-parameters"""
        return deepcopy(self._hyper["model"])

    def data(self):
        """Return copy of data section of hyper-parameters"""
        return deepcopy(self._hyper["data"])

    def training(self):
        """Return copy of training section of hyper-parameters"""
        return deepcopy(self._hyper["training"])

    def inputs(self):
        """Return copy of model/inputs section of hyper-parameters"""
        return deepcopy(self._hyper["model"]["inputs"])

    def dataset(self):
        """Return copy of data/dataset section of hyper-parameters"""
        return deepcopy(self._hyper["data"]["dataset"])

    def compile(self, loss=None, optimizer='rmsprop', metrics: list = None, weighted_metrics: list = None):
        """Select compile hyper-parameter. Additional default values for the training scripts are given as
        functional kwargs. Functional kwargs are overwritten by hyper-parameters.

        Args:
            loss: Default loss for fit. Default is None.
            optimizer: Default optimizer. Default is "rmsprop".
            metrics (list): Default list of metrics. Default is None.
            weighted_metrics (list): Default list of weighted_metrics. Default is None.

        Returns:
            dict: de-serialized compile kwargs from hyper-parameters.
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
        """Select fit hyper-parameter. Additional default values for the training scripts are given as
        functional kwargs. Functional kwargs are overwritten by hyper-parameters.

        Args:
            epochs (int): Default number of epochs. Default is 1.
            validation_freq (int): Default validation frequency. Default is 1.
            batch_size (int): Default batch size. Default is None.
            callbacks (list): Default Callbacks. Default is None.

        Returns:
            dict: de-serialized fit kwargs from hyper-parameters.
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

    def cross_validation(self):
        """Configuration for cross-validation. At the moment no de-serialization is used, but the hyper-parameters
        can be set with keras-like serialization.

        Returns:
            dict: Cross-validation configuration from hyper-parameters.
        """
        return self.training()["cross_validation"]

    def results_file_path(self):
        r"""Make output folder for results based on hyper-parameter and return path to that folder.
        The folder is set up as `'results'/dataset/model_name + post_fix`. Where model and dataset name must be set by
        this class. Postfix can be in hyper-parameters setting.

        Returns:
            str: File-path or path object to result folder.
        """
        hyper_info = deepcopy(self._hyper["info"])
        post_fix = str(hyper_info["postfix"]) if "postfix" in hyper_info else ""
        os.makedirs("results", exist_ok=True)
        os.makedirs(os.path.join("results", self.dataset_name), exist_ok=True)
        model_name = self._hyper['model']['name']
        filepath = os.path.join("results", self.dataset_name, model_name + post_fix)
        os.makedirs(filepath, exist_ok=True)
        return filepath

    def postfix_file(self):
        """Return a postfix for naming files in the fit results section.

        Returns:
            str: Postfix for naming output files.
        """
        hyper_info = deepcopy(self._hyper["info"])
        return str(hyper_info["postfix_file"]) if "postfix_file" in hyper_info else ""

    def save(self, file_path: str):
        """Save the hyper-parameter to path.

        Args:
            file_path (str): Full file path to save hyper-parameters to.
        """
        # Must make more refined saving and serialization here.
        save_json_file(self._hyper, file_path)

    def make_model(self):
        r"""Dictionary of model kwargs. Can be nested.

        Returns:
            dict: Model kwargs for :obj:`make_model()` function from hyper-parameters.
        """
        return deepcopy(self._hyper["model"])

    def use_scaler(self, use_scaler=False):
        r"""Use scaler. Functional kwargs are overwritten by hyper-parameters.

        Args:
            use_scaler (bool): Whether to use scaler as default. Default is False.

        Returns:
            bool: Flag for using scaler from hyper-parameters.
        """
        if "scaler" in self.training():
            return True
        return use_scaler

    def data_unit(self, data_unit=""):
        """Optional unit for targets or the data in general. Functional kwargs are overwritten by hyper-parameters.
        Alternatively the units can be defined by the dataset directly. Mostly relevant for plotting or annotate stats.

        Args:
            data_unit (str): Default unit for the dataset. Default is "".

        Returns:
            str: Units for the data from hyper-parameters.
        """
        if "data_unit" in self.data():
            return self.data()["data_unit"]
        return data_unit

    def scaler(self):
        """Configuration for standard scaler. At the moment no de-serialization is used, but the hyper-parameters
        can be set with keras-like serialization.

        Returns:
            dict: Cross-validation configuration from hyper-parameters.
        """
        return self.training()["scaler"]

    def k_fold(self, n_splits: int = 5, shuffle: bool = None, random_state: int = None):
        """Select k-fold hyper-parameter. Not used anymore for current training scripts. Functional kwargs
        are overwritten by hyper-parameters.

        Args:
            n_splits (int): Default number of splits. Default is 5.
            shuffle (bool): Shuffle data by default. Default is None.
            random_state (int): Default random state for shuffle. Default is None.

        Returns:
            dict: Dictionary of kwargs for sklearn KFold() class from hyper-parameters.
        """
        k_fold_info = {"n_splits": n_splits, "shuffle": shuffle, "random_state": random_state}
        if "KFold" in self._hyper["training"]:
            k_fold_info.update(self._hyper["training"]["KFold"])
        return k_fold_info

    def execute_splits(self, execute_splits=np.inf):
        """The number of splits to execute apart form the settings in cross-validation. Could be used if not all
        splits should be fitted for cost reasons. Functional kwargs are overwritten by hyper-parameters.

        Args:
            execute_splits (int): The default number of splits. Default is inf.

        Returns:
            int: The number of splits to run from hyper-parameters.
        """
        if "execute_splits" in self.training():
            return int(self.training()["execute_splits"])
        if "execute_folds" in self.training():
            return int(self.training()["execute_folds"])
        return execute_splits

    def multi_target_indices(self, multi_target_indices=None):
        """Get list of target indices for multi-target training. Functional kwargs are overwritten by hyper-parameters.

        Args:
            multi_target_indices (list): T

        Returns:
            list: List of target indices from hyper-parameters.
        """
        if "multi_target_indices" in self.training():
            return self.training()["multi_target_indices"]
        return multi_target_indices


# Only for backward compatibility.
HyperSelectionTraining = HyperSelection

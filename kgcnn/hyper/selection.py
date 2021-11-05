import tensorflow as tf

from copy import deepcopy
from kgcnn.utils.data import load_hyper_file


class HyperSelection:
    r"""A class to store hyper-parameters for a specific dataset and model, exposing them for model training scripts.
    This includes training parameters and a set of general information like a path for output of the training stats
    or the expected version of `kgcnn`. The class methods will extract and possibly serialize or deserialized the
    necessary kwargs from the hyper-parameter dictionary with additional default values. Also changes in the
    hyper-parameter definition can be made without affecting training scripts and made compatible with previous
    versions.


    """

    def __init__(self, hyper_info: str, model_name: str = None, dataset_name: str = None):
        """Make a hyper-parameter class instance.

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
            raise ValueError("ERROR:kgcnn: `HyperSelection` requires valid hyper dictionary or path to file.")

        self._hyper = None
        if "model" in self._hyper_all and "training" in self._hyper_all:
            self._hyper = self._hyper_all
        elif model_name is not None:
            self._hyper = self._hyper_all[model_name]
        else:
            raise ValueError("ERROR:kgcnn: Not a valid hyper dictionary. Please provide model_name.")

    def hyper(self, section=None):
        """Get copy of the hyper-parameter dictionary stored by this class.

        Args:
            section (str): If specified, return copy of hyper[selection].

        Returns:
            dict: Hyper-parameters
        """
        if section is None:
            return deepcopy(self._hyper)
        else:
            return deepcopy(self._hyper[section])

    def get_hyper(self, section=None):
        # Only for backward compatibility.
        return self.hyper(section)

    def compile(self, loss=None, optimizer='rmsprop', metrics=None, weighted_metrics=None):
        """Select compile hyper-parameter.

        Args:
            loss: Loss for fit.
            optimizer: Optimizer.
            metrics (list): List of metrics
            weighted_metrics (list): List of weighted_metrics

        Returns:
            dict: de-serialized hyper-parameter
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

    def fit(self, epochs=1, validation_freq=1, batch_size=None, callbacks=None):
        """Select fit hyper-parameter.

        Args:
            epochs (int): Number of epochs.
            validation_freq (int): Validation frequency.
            batch_size (int): Batch size.
            callbacks (list): Callbacks

        Returns:
            dict: de-serialized hyper-parameters.
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

    def k_fold(self, n_splits: int = 5, shuffle: bool = None, random_state: int = None):
        """Select k-fold hyper-parameter.

        Args:
            n_splits (int): Number of splits
            shuffle (bool): Shuffle data.
            random_state (int): Random state for shuffle.

        Returns:
            dict: Dictionary of kwargs for sklearn KFold() class.
        """
        k_fold_info = {"n_splits": n_splits, "shuffle": shuffle, "random_state": random_state}
        if "KFold" in self._hyper["training"]:
            k_fold_info.update(self._hyper["training"]["KFold"])
        return k_fold_info


# Only for backward compatibility.
HyperSelectionTraining = HyperSelection

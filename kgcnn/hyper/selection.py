import tensorflow as tf

from copy import deepcopy
from kgcnn.utils.data import load_json_file, load_hyper_file


class HyperSelectionTraining:
    r"""A class to choose a hyper-parameters for a specific dataset and model. And also to serialize hyper info,
    if possible. Will be added soon.

    """

    def __init__(self, path: str, model_name: str = None, dataset_name: str = None):
        self.dataset_name = dataset_name
        self._hyper_all = load_hyper_file(path)
        self._hyper = None
        if "model" in self._hyper_all and "training" in self._hyper_all:
            self._hyper = self._hyper_all
        elif model_name is not None:
            self._hyper = self._hyper_all[model_name]
        else:
            raise ValueError("ERROR:kgcnn: Not a valid hyper dictionary. Please provide model_name.")

    def get_hyper(self, section = None):
        if section is None:
            return deepcopy(self._hyper)
        else:
            return deepcopy(self._hyper[section])

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



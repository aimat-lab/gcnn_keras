import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LinearWarmupExponentialLearningRateScheduler')
class LinearWarmupExponentialLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    """Callback for linear change of the learning rate. This class inherits from
    tf.keras.callbacks.LearningRateScheduler."""

    def __init__(self, lr_start: float, decay_gamma: float, epo_warmup: int = 10, lr_min: float = 0.0,
                 verbose: int = 0):
        """Set the parameters for the learning rate scheduler.

        Args:
            lr_start (float): Learning rate at the start of the exp. decay.
            decay_gamma (float): Gamma parameter in the exponential.
            epo_warmup (int): Number of warm-up steps. Default is 10.
            lr_min (float): Minimum learning rate allowed during the decay. Default is 0.0.
            verbose (int): Verbosity. Default is 0.
        """
        self.decay_gamma = decay_gamma
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.epo_warmup = max(epo_warmup, 0)
        self.verbose = verbose
        super(LinearWarmupExponentialLearningRateScheduler, self).__init__(schedule=self.schedule_epoch_lr,
                                                                           verbose=verbose)

    def schedule_epoch_lr(self, epoch, lr):
        """Reduce the learning rate."""
        if epoch < self.epo_warmup:
            new_lr = self.lr_start * epoch / self.epo_warmup + self.lr_min
        elif epoch == self.epo_warmup:
            new_lr = max(self.lr_start, self.lr_min)
        else:
            new_lr = max(self.lr_start * np.exp(-(epoch - self.epo_warmup) / self.decay_gamma), self.lr_min)
        return float(new_lr)

    def get_config(self):
        config = super(LinearWarmupExponentialLearningRateScheduler, self).get_config()
        config.update({"lr_start": self.lr_start, "decay_gamma": self.decay_gamma, "epo_warmup": self.epo_warmup,
                       "lr_min": self.lr_min, "verbose": self.verbose})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LinearLearningRateScheduler')
class LinearLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    """Callback for linear change of the learning rate. This class inherits from
    tf.keras.callbacks.LearningRateScheduler."""

    def __init__(self, learning_rate_start: float = 1e-3, learning_rate_stop: float = 1e-5, epo_min: int = 0,
                 epo: int = 500, verbose: int = 0, eps: float = 1e-8):
        """Set the parameters for the learning rate scheduler.

        Args:
            learning_rate_start (float): Initial learning rate. Default is 1e-3.
            learning_rate_stop (float): End learning rate. Default is 1e-5.
            epo_min (int): Minimum number of epochs to keep the learning-rate constant before decrease. Default is 0.
            epo (int): Total number of epochs. Default is 500.
            eps (float): Numerical epsilon which bounds minimum learning rate.
            verbose (int): Verbosity. Default is 0.
        """
        super(LinearLearningRateScheduler, self).__init__(schedule=self.schedule_epoch_lr, verbose=verbose)
        self.learning_rate_start = learning_rate_start
        self.learning_rate_stop = learning_rate_stop
        self.epo = epo
        self.epo_min = epo_min
        self.eps = float(eps)

    def schedule_epoch_lr(self, epoch, lr):
        if epoch < self.epo_min:
            out = float(self.learning_rate_start)
        else:
            out = float(self.learning_rate_start - (self.learning_rate_start - self.learning_rate_stop) / (
                    self.epo - self.epo_min) * (epoch - self.epo_min))
        return max(float(out), self.eps)

    def get_config(self):
        config = super(LinearLearningRateScheduler, self).get_config()
        config.update({"learning_rate_start": self.learning_rate_start,
                       "learning_rate_stop": self.learning_rate_stop,
                       "epo": self.epo, "epo_min": self.epo_min, "eps": self.eps})
        return config
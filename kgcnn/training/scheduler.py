import numpy as np
import tensorflow as tf
import math
ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='CosineAnnealingLR')
class CosineAnnealingLR(ks.callbacks.LearningRateScheduler):
    r"""Callback for exponential learning rate schedule with warmup. This class inherits from
    ks.callbacks.LearningRateScheduler."""

    def __init__(self, lr_start: float, epoch_max: int, lr_min: float = 0, verbose: int = 0):
        """Set the parameters for the learning rate scheduler.

        Args:
            lr_start (float): Learning rate at the start of the annealing. decay.
            epoch_max (int): Maximum number of iterations.
            lr_min (float): Minimum learning rate allowed during the decay. Default is 0.0.
            verbose (int): Verbosity. Default is 0.
        """
        self.epoch_max = max(epoch_max, 0)
        self.lr_min = lr_min
        self.lr_start = lr_start
        self.verbose = verbose
        super(CosineAnnealingLR, self).__init__(
            schedule=self.schedule_epoch_lr, verbose=verbose)

    def schedule_epoch_lr(self, epoch, lr):
        """Reduce the learning rate."""
        new_lr = self.lr_min + (self.lr_start - self.lr_min) * (
                1 + math.cos(math.pi * epoch / self.epoch_max)) / 2
        return float(new_lr)

    def get_config(self):
        config = super(CosineAnnealingLR, self).get_config()
        config.update({"lr_start": self.lr_start, "epoch_max": self.epoch_max,
                       "lr_min": self.lr_min, "verbose": self.verbose})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='LinearWarmupExponentialLRScheduler')
class LinearWarmupExponentialLRScheduler(ks.callbacks.LearningRateScheduler):
    r"""Callback for exponential learning rate schedule with warmup. This class inherits from
    ks.callbacks.LearningRateScheduler."""

    def __init__(self, lr_start: float, gamma: float, epo_warmup: int = 10, lr_min: float = 0.0,
                 verbose: int = 0):
        """Set the parameters for the learning rate scheduler.

        Args:
            lr_start (float): Learning rate at the start of the exp. decay.
            gamma (float): Multiplicative factor of learning rate decay.
            epo_warmup (int): Number of warm-up epochs. Default is 10.
            lr_min (float): Minimum learning rate allowed during the decay. Default is 0.0.
            verbose (int): Verbosity. Default is 0.
        """
        self.gamma = gamma
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.epo_warmup = max(epo_warmup, 0)
        self.verbose = verbose
        super(LinearWarmupExponentialLRScheduler, self).__init__(
            schedule=self.schedule_epoch_lr, verbose=verbose)

    def schedule_epoch_lr(self, epoch, lr):
        """Reduce the learning rate."""
        if epoch < self.epo_warmup:
            new_lr = self.lr_start * epoch / self.epo_warmup + self.lr_min
        elif epoch == self.epo_warmup:
            new_lr = max(self.lr_start, self.lr_min)
        else:
            new_lr = max(self.lr_start * self.gamma ** (epoch - self.epo_warmup), self.lr_min)
        return float(new_lr)

    def get_config(self):
        config = super(LinearWarmupExponentialLRScheduler, self).get_config()
        config.update({"lr_start": self.lr_start, "gamma": self.gamma, "epo_warmup": self.epo_warmup,
                       "lr_min": self.lr_min, "verbose": self.verbose})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='LinearWarmupExponentialLearningRateScheduler')
class LinearWarmupExponentialLearningRateScheduler(ks.callbacks.LearningRateScheduler):
    r"""Callback for exponential learning rate schedule with warmup. This class inherits from
    ks.callbacks.LearningRateScheduler."""

    def __init__(self, lr_start: float, decay_lifetime: float, epo_warmup: int = 10, lr_min: float = 0.0,
                 verbose: int = 0):
        """Set the parameters for the learning rate scheduler.

        Args:
            lr_start (float): Learning rate at the start of the exp. decay.
            decay_lifetime (float): Tau parameter in the exponential for epochs.
            epo_warmup (int): Number of warm-up epochs. Default is 10.
            lr_min (float): Minimum learning rate allowed during the decay. Default is 0.0.
            verbose (int): Verbosity. Default is 0.
        """
        self.decay_lifetime = decay_lifetime
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.epo_warmup = max(epo_warmup, 0)
        self.verbose = verbose
        super(LinearWarmupExponentialLearningRateScheduler, self).__init__(
            schedule=self.schedule_epoch_lr, verbose=verbose)

    def schedule_epoch_lr(self, epoch, lr):
        """Reduce the learning rate."""
        if epoch < self.epo_warmup:
            new_lr = self.lr_start * epoch / self.epo_warmup + self.lr_min
        elif epoch == self.epo_warmup:
            new_lr = max(self.lr_start, self.lr_min)
        else:
            new_lr = max(self.lr_start * np.exp(-(epoch - self.epo_warmup) / self.decay_lifetime), self.lr_min)
        return float(new_lr)

    def get_config(self):
        config = super(LinearWarmupExponentialLearningRateScheduler, self).get_config()
        config.update({"lr_start": self.lr_start, "decay_lifetime": self.decay_lifetime, "epo_warmup": self.epo_warmup,
                       "lr_min": self.lr_min, "verbose": self.verbose})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='LinearLearningRateScheduler')
class LinearLearningRateScheduler(ks.callbacks.LearningRateScheduler):
    """Callback for linear change of the learning rate. This class inherits from
    ks.callbacks.LearningRateScheduler."""

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


@ks.utils.register_keras_serializable(package='kgcnn', name='LinearWarmupLinearLearningRateScheduler')
class LinearWarmupLinearLearningRateScheduler(ks.callbacks.LearningRateScheduler):
    """Callback for linear change of the learning rate. This class inherits from
    ks.callbacks.LearningRateScheduler."""

    def __init__(self, learning_rate_start: float = 1e-3, learning_rate_stop: float = 1e-5, epo_warmup: int = 0,
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
        super(LinearWarmupLinearLearningRateScheduler, self).__init__(schedule=self.schedule_epoch_lr, verbose=verbose)
        self.learning_rate_start = learning_rate_start
        self.learning_rate_stop = learning_rate_stop
        self.epo = epo
        self.epo_warmup = epo_warmup
        self.eps = float(eps)

    def schedule_epoch_lr(self, epoch, lr):
        if epoch < self.epo_warmup:
            out = self.learning_rate_start*epoch/self.epo_warmup
        else:
            out = self.learning_rate_start - (self.learning_rate_start - self.learning_rate_stop) / (
                    self.epo - self.epo_warmup) * (epoch - self.epo_warmup)
        return max(float(out), self.eps)

    def get_config(self):
        config = super(LinearWarmupLinearLearningRateScheduler, self).get_config()
        config.update({"learning_rate_start": self.learning_rate_start,
                       "learning_rate_stop": self.learning_rate_stop,
                       "epo": self.epo, "epo_warmup": self.epo_warmup, "eps": self.eps})
        return config

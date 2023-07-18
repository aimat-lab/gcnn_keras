import numpy as np
import tensorflow as tf
import math
import logging
ks = tf.keras

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


class LinearWarmUpScheduler(ks.callbacks.LearningRateScheduler):
    r"""Warmup scheduler that increases the learning rate over the first steps or epochs.
    Inherits from :obj:`ks.callbacks.LearningRateScheduler`. Note that the parameter :obj:`schedule` in the
    constructor can be `None` which sets the wam-up in :obj:`linear_warmup_schedule_epoch_lr` only.
    Another learning rate schedule can be provided that makes use of :obj:`linear_warmup_schedule_epoch_lr` in
    subclasses.

    .. note::

        To increase the learning rate within each epoch, you must provide :obj:`steps_per_epoch`.
    """

    def __init__(self, schedule=None, verbose: int = 0, steps_per_epoch: int = None,
                 lr_start: float = None, epo_warmup: int = 0):
        r"""Initialize class.

        Args:
            schedule (Callable): Learning rate schedule. Default is None.
            verbose (int): Verbosity. Default is 0.
            steps_per_epoch (int): Steps per epoch. Required for sub-epoch increase. Default is None.
            lr_start (float): Learning rate to reach in linear ramp to start learning. Default is None.
            epo_warmup (int): Number of warmup epochs. Default is None.
        """
        # This is for passing other schedule through warmup to `ks.callbacks.LearningRateScheduler`.
        # IDEA: We could pass a restart option.
        if schedule is None:
            schedule = self.linear_warmup_schedule_epoch_lr
        self.verbose = verbose
        super(LinearWarmUpScheduler, self).__init__(
            schedule=schedule, verbose=verbose)
        self.epo_warmup = max(epo_warmup, 0)
        self.__warming_up = False
        self.steps_per_epoch = steps_per_epoch
        self.lr_start = lr_start
        if self.steps_per_epoch is None:
            module_logger.warning("`steps_per_epoch` is not set. Can't increase lr during epochs of warmup.")

    def linear_warmup_schedule_epoch_lr(self, epoch, lr):
        """Learning rate schedule function for warmup.

        Returns the current learning rate unchanged apart from warmup period.

        Args:
            epoch (int): Epoch index (integer, indexed from 0).
            lr (float): Current learning rate.

        Returns:
            float: New learning rate.
        """
        if epoch < self.epo_warmup:
            self.__warming_up = True
            new_lr = float(self.lr_start * epoch / self.epo_warmup)
        else:
            self.__warming_up = False
            new_lr = lr
        return new_lr

    def on_train_batch_begin(self, batch, logs=None):
        """Recursive increase of the learning rate warmup between epochs.

        Args:
            batch (int): Batch index (integer, indexed from 0).
            logs: Not used.

        Returns:
            None
        """
        if self.__warming_up and self.steps_per_epoch is not None:
            if not hasattr(self.model.optimizer, "lr"):
                raise ValueError('Optimizer must have a "lr" attribute.')
            lr = float(ks.backend.get_value(self.model.optimizer.lr))
            lr = lr + self.lr_start / self.epo_warmup / self.steps_per_epoch
            if batch > self.steps_per_epoch:
                module_logger.warning("Found `batch` > `steps_per_epoch` during warmup.")
            if self.verbose > 0:
                print("{0}/{1}: Warmup-step increase lr to {2}.".format(batch, self.steps_per_epoch, lr))
            ks.backend.set_value(self.model.optimizer.lr, ks.backend.get_value(lr))

    def get_config(self):
        """Get config for this class."""
        config = super(LinearWarmUpScheduler, self).get_config()
        config.update({"lr_start": self.lr_start, "epo_warmup": self.epo_warmup,
                       "verbose": self.verbose, "steps_per_epoch": self.steps_per_epoch})
        return config

    @classmethod
    def from_config(cls, config):
        """Make class instance from config."""
        return cls(**config)


@ks.utils.register_keras_serializable(package='kgcnn', name='CosineAnnealingLRScheduler')
class CosineAnnealingLRScheduler(ks.callbacks.LearningRateScheduler):
    r"""Callback for cosine learning rate (LR) schedule. This class inherits from
    :obj:`ks.callbacks.LearningRateScheduler` and applies :obj:`schedule_epoch_lr`.
    Proposed by `SGDR <https://arxiv.org/abs/1608.03983>`_.
    The cosine part without restarts for the LR Schedule follows:

    .. math::

        \eta_t (T_{cur}) = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)
    """

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
        super(CosineAnnealingLRScheduler, self).__init__(
            schedule=self.schedule_epoch_lr, verbose=verbose)

    def schedule_epoch_lr(self, epoch, lr):
        """Closed from of learning rate."""
        new_lr = self.lr_min + (self.lr_start - self.lr_min) * (
                1 + math.cos(math.pi * epoch / self.epoch_max)) / 2
        return float(new_lr)

    def get_config(self):
        """Get config for this class."""
        config = super(CosineAnnealingLRScheduler, self).get_config()
        config.update({"lr_start": self.lr_start, "epoch_max": self.epoch_max,
                       "lr_min": self.lr_min, "verbose": self.verbose})
        return config

    @classmethod
    def from_config(cls, config):
        """Make class instance from config."""
        return cls(**config)


@ks.utils.register_keras_serializable(package='kgcnn', name='LinearWarmupExponentialLRScheduler')
class LinearWarmupExponentialLRScheduler(LinearWarmUpScheduler):
    r"""Callback for exponential learning rate schedule with warmup. This class inherits from
    :obj:`LinearWarmUpScheduler`, which inherits from :obj:`ks.callbacks.LearningRateScheduler`.
    The learning rate :math:`\eta` is reduced or increased (usually :math:`\gamma < 1`) after
    warmup epochs :math:`T_0` as:

    .. math::

        \eta (T) = \eta_0 \; \gamma^{T-T_0}
    """

    def __init__(self, lr_start: float, gamma: float, epo_warmup: int = 10, lr_min: float = 0.0,
                 verbose: int = 0, steps_per_epoch: int = None):
        """Set the parameters for the learning rate scheduler.

        Args:
            lr_start (float): Learning rate at the start of the exponential decay.
            gamma (float): Multiplicative factor of learning rate decay.
            epo_warmup (int): Number of warmup epochs. Default is 10.
            lr_min (float): Minimum learning rate allowed during the decay and start. Default is 0.0.
            verbose (int): Verbosity. Default is 0.
            steps_per_epoch (int): Number of steps per epoch. Required for warmup to linearly increase between epochs.
                Default is None.
        """
        self.gamma = gamma
        self.lr_min = lr_min
        super(LinearWarmupExponentialLRScheduler, self).__init__(
            schedule=self.schedule_epoch_lr, verbose=verbose, lr_start=lr_start, epo_warmup=epo_warmup,
            steps_per_epoch=steps_per_epoch)

    def schedule_epoch_lr(self, epoch, lr):
        """Change the learning rate after warmup exponentially.

        Args:
            epoch (int): Epoch index (integer, indexed from 0).
            lr (float): Current learning rate. Not used.

        Returns:
            float: New learning rate.
        """
        new_lr = self.lr_start * (self.gamma ** (epoch - self.epo_warmup))
        new_lr = self.linear_warmup_schedule_epoch_lr(epoch, new_lr)
        return max(float(new_lr), self.lr_min)

    def get_config(self):
        """Get config for this class."""
        config = super(LinearWarmupExponentialLRScheduler, self).get_config()
        config.update({"gamma": self.gamma, "lr_min": self.lr_min})
        return config

    @classmethod
    def from_config(cls, config):
        """Make class instance from config."""
        return cls(**config)


@ks.utils.register_keras_serializable(package='kgcnn', name='LinearWarmupExponentialLearningRateScheduler')
class LinearWarmupExponentialLearningRateScheduler(LinearWarmUpScheduler):
    r"""Callback for exponential learning rate schedule with warmup. This class inherits from
    :obj:`LinearWarmUpScheduler`, which inherits from :obj:`ks.callbacks.LearningRateScheduler`.
    The learning rate :math:`\eta` is reduced after warmup epochs :math:`T_0` with lifetime :math:`\tau` as:

    .. math::

        \eta (T) = \eta_0 \; e^{-\frac{T-T_0}{\tau}}
    """

    def __init__(self, lr_start: float, decay_lifetime: float, epo_warmup: int = 10, lr_min: float = 0.0,
                 verbose: int = 0, steps_per_epoch: int = None):
        """Set the parameters for the learning rate scheduler.

        Args:
            lr_start (float): Learning rate at the start of the exp. decay.
            decay_lifetime (float): Tau or lifetime parameter in the exponential in epochs.
            epo_warmup (int): Number of warmup epochs. Default is 10.
            lr_min (float): Minimum learning rate allowed during the decay. Default is 0.0.
            verbose (int): Verbosity. Default is 0.
            steps_per_epoch (int): Number of steps per epoch. Required for warmup to linearly increase between epochs.
                Default is None.
        """
        self.decay_lifetime = decay_lifetime
        self.lr_min = lr_min
        super(LinearWarmupExponentialLearningRateScheduler, self).__init__(
            schedule=self.schedule_epoch_lr, verbose=verbose, lr_start=lr_start, epo_warmup=epo_warmup,
            steps_per_epoch=steps_per_epoch)

    def schedule_epoch_lr(self, epoch, lr):
        """Reduce the learning rate after warmup exponentially.

        Args:
            epoch (int): Epoch index (integer, indexed from 0).
            lr (float): Current learning rate. Not used.

        Returns:
            float: New learning rate.
        """
        new_lr = float(self.lr_start * np.exp(-(epoch - self.epo_warmup) / self.decay_lifetime))
        new_lr = self.linear_warmup_schedule_epoch_lr(epoch, new_lr)
        return max(new_lr, self.lr_min)

    def get_config(self):
        """Get config for this class."""
        config = super(LinearWarmupExponentialLearningRateScheduler, self).get_config()
        config.update({"decay_lifetime": self.decay_lifetime, "lr_min": self.lr_min})
        return config

    @classmethod
    def from_config(cls, config):
        """Make class instance from config."""
        return cls(**config)


@ks.utils.register_keras_serializable(package='kgcnn', name='LinearLearningRateScheduler')
class LinearLearningRateScheduler(ks.callbacks.LearningRateScheduler):
    r"""Callback for linear change of the learning rate. This class inherits from
    :obj:`ks.callbacks.LearningRateScheduler`.
    The learning rate :math:`\eta_0` is reduced linearly at :math:`T_0` epochs up to :math:`T_{max}`
    epochs to reach the learning rate :math:`\eta_{T_{max}}`:

    .. math::

        \eta (T) = \eta_0 - \frac{\eta_0 - \eta_{T_{max}}}{T_{max}-T_0} (T-T_0)  \;\; \text{for} \;\; T>T_0
    """

    def __init__(self, learning_rate_start: float = 1e-3, learning_rate_stop: float = 1e-5, epo_min: int = 0,
                 epo: int = 500, verbose: int = 0, eps: float = 1e-8):
        """Set the parameters for the learning rate scheduler.

        Args:
            learning_rate_start (float): Initial learning rate. Default is 1e-3.
            learning_rate_stop (float): End learning rate. Default is 1e-5.
            epo_min (int): Minimum number of epochs to keep the learning-rate constant before decrease. Default is 0.
            epo (int): Total number of epochs. Default is 500.
            eps (float): Minimum learning rate. Default is 1e-08.
            verbose (int): Verbosity. Default is 0.
        """
        super(LinearLearningRateScheduler, self).__init__(schedule=self.schedule_epoch_lr, verbose=verbose)
        self.learning_rate_start = learning_rate_start
        self.learning_rate_stop = learning_rate_stop
        self.epo = epo
        self.epo_min = epo_min
        self.eps = float(eps)

    def schedule_epoch_lr(self, epoch, lr):
        r"""Reduce the learning linearly.

        Kept constant for :obj:`epo_min` before decrease.

        Args:
            epoch (int): Epoch index (integer, indexed from 0).
            lr (float): Current learning rate. Not used.

        Returns:
            float: New learning rate.
        """
        if epoch < self.epo_min:
            out = float(self.learning_rate_start)
        else:
            out = float(self.learning_rate_start - (self.learning_rate_start - self.learning_rate_stop) / (
                    self.epo - self.epo_min) * (epoch - self.epo_min))
        return max(float(out), self.eps)

    def get_config(self):
        """Get config for this class."""
        config = super(LinearLearningRateScheduler, self).get_config()
        config.update({"learning_rate_start": self.learning_rate_start,
                       "learning_rate_stop": self.learning_rate_stop,
                       "epo": self.epo, "epo_min": self.epo_min, "eps": self.eps})
        return config

    @classmethod
    def from_config(cls, config):
        """Make class instance from config."""
        return cls(**config)


@ks.utils.register_keras_serializable(package='kgcnn', name='LinearWarmupLinearLearningRateScheduler')
class LinearWarmupLinearLearningRateScheduler(LinearWarmUpScheduler):
    r"""Callback for linear change of the learning rate with warmup. This class inherits from
    :obj:`LinearWarmUpScheduler`, which inherits from :obj:`ks.callbacks.LearningRateScheduler`.
    The learning rate is increased in a warmup phase of :math:`T_0` epochs up to :math:`\eta_0`.
    The learning rate :math:`\eta_0` is reduced linearly at :math:`T_0` epochs up to :math:`T_{max}`
    epochs to reach the learning rate :math:`\eta_{T_{max}}`:

    .. math::

        \eta (T) = \eta_0 - \frac{\eta_0 - \eta_{T_{max}}}{T_{max}-T_0} (T-T_0)  \;\; \text{for} \;\; T>T_0
    """

    def __init__(self, learning_rate_start: float = 1e-3, learning_rate_stop: float = 1e-5, epo_warmup: int = 0,
                 epo: int = 500, verbose: int = 0, eps: float = 1e-8, steps_per_epoch: int = None,
                 lr_start: int = None):
        """Set the parameters for the learning rate scheduler.

        Args:
            learning_rate_start (float): Initial learning rate. Default is 1e-3.
            learning_rate_stop (float): End learning rate. Default is 1e-5.
            epo (int): Total number of epochs. Default is 500.
            eps (float): Minimum learning rate. Default is 1e-08.
            verbose (int): Verbosity. Default is 0.
            steps_per_epoch (int): Number of steps per epoch. Required for warmup to linearly increase between epochs.
                Default is None.
            lr_start (int): Ignored, set to `learning_rate_start`.
        """
        super(LinearWarmupLinearLearningRateScheduler, self).__init__(
            schedule=self.schedule_epoch_lr, verbose=verbose, lr_start=learning_rate_start, epo_warmup=epo_warmup,
            steps_per_epoch=steps_per_epoch)
        self.learning_rate_start = learning_rate_start
        self.learning_rate_stop = learning_rate_stop
        self.epo = epo
        self.eps = float(eps)

    def schedule_epoch_lr(self, epoch, lr):
        """Reduce the learning linearly after warmup.

        Args:
            epoch (int): Epoch index (integer, indexed from 0).
            lr (float): Current learning rate. Not used.

        Returns:
            float: New learning rate.
        """
        new_lr = self.learning_rate_start - (self.learning_rate_start - self.learning_rate_stop) / (
                self.epo - self.epo_warmup) * (epoch - self.epo_warmup)
        new_lr = self.linear_warmup_schedule_epoch_lr(epoch, new_lr)
        return max(float(new_lr), self.eps)

    def get_config(self):
        """Get config for this class."""
        config = super(LinearWarmupLinearLearningRateScheduler, self).get_config()
        config.update({
            "learning_rate_start": self.learning_rate_start, "learning_rate_stop": self.learning_rate_stop,
            "epo": self.epo, "eps": self.eps})
        return config

    @classmethod
    def from_config(cls, config):
        """Make class instance from config."""
        return cls(**config)


@ks.utils.register_keras_serializable(package='kgcnn', name='PolynomialDecayScheduler')
class PolynomialDecayScheduler(ks.callbacks.LearningRateScheduler):
    r"""Callback for polynomial decay of the learning rate. This class inherits from
    :obj:`ks.callbacks.LearningRateScheduler`.

    Adapts :obj:`keras.optimizers.schedules.PolynomialDecay` to a scheduler with a function of epochs.

    """

    def __init__(self, initial_learning_rate, decay_epochs, end_learning_rate=0.0001, power=1.0, cycle=False,
                 verbose: int = 0, eps: float = 1e-8):
        """Set the parameters for the learning rate scheduler.

        Args:
            initial_learning_rate (float): The initial learning rate
            decay_epochs (float): Must be positive. See the decay computation above.
            end_learning_rate (float): The minimal end learning rate.
            power (float): The power of the polynomial. Defaults to `1.0` .
            cycle (bool): A boolean, whether it should cycle beyond decay_steps.
            eps (float): Minimum learning rate. Default is 1e-08.
            verbose (int): Verbosity. Default is 0.
        """
        super(PolynomialDecayScheduler, self).__init__(schedule=self.schedule_epoch_lr, verbose=verbose)
        self.initial_learning_rate = initial_learning_rate
        self.decay_epochs = decay_epochs
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        self.verbose = verbose
        self.eps = eps

    def schedule_epoch_lr(self, epoch, lr):
        r"""Reduce the learning linearly.

        Args:
            epoch (int): Epoch index (integer, indexed from 0).
            lr (float): Current learning rate. Not used.

        Returns:
            float: New learning rate.
        """
        if not self.cycle:
            epoch = np.minimum(self.decay_epochs, epoch)
            decay_epoch = self.decay_epochs
        else:
            decay_epoch = self.decay_epochs * np.ceil(epoch / self.decay_epochs)
        pp = np.power(1 - epoch / decay_epoch, self.power)
        new_lr = (self.initial_learning_rate - self.end_learning_rate) * pp + self.end_learning_rate
        return max(float(new_lr), float(self.eps))

    def get_config(self):
        """Get config for this class."""
        config = super(PolynomialDecayScheduler, self).get_config()
        config.update({
            "initial_learning_rate": self.initial_learning_rate,
            "decay_epochs": self.decay_epochs,
            "end_learning_rate": self.end_learning_rate,
            "power": self.power,
            "cycle": self.cycle,
            "verbose": self.verbose,
            "eps": self.eps
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Make class instance from config."""
        return cls(**config)
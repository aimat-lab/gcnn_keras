import numpy as np
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LinearWarmupExponentialLearningRateScheduler')
class LinearWarmupExponentialLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    """Callback for linear change of the learning rate."""

    def __init__(self, decay_rate, verbose=0):
        self.decay_rate = decay_rate
        super(LinearWarmupExponentialLearningRateScheduler, self).__init__(schedule=self.schedule_implement,
                                                                           verbose=verbose)

    def schedule_implement(self, epoch, lr):
        out = self.decay_rate*lr
        return float(out)

    def get_config(self):
        config = super(LinearWarmupExponentialLearningRateScheduler, self).get_config()
        config.update({})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LinearLearningRateScheduler')
class LinearLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    """Callback for linear change of the learning rate."""

    def __init__(self, learning_rate_start=1e-3, learning_rate_stop=1e-5, epo_min=0, epo=500, verbose=0):
        super(LinearLearningRateScheduler, self).__init__(schedule=self.schedule_implement, verbose=verbose)
        self.learning_rate_start = learning_rate_start
        self.learning_rate_stop = learning_rate_stop
        self.epo = epo
        self.epo_min = epo_min

    def schedule_implement(self, epoch, lr):
        if epoch < self.epo_min:
            out = float(self.learning_rate_start)
        else:
            out = float(self.learning_rate_start - (self.learning_rate_start - self.learning_rate_stop) / (
                        self.epo - self.epo_min) * (epoch - self.epo_min))
        return float(out)

    def get_config(self):
        config = super(LinearLearningRateScheduler, self).get_config()
        config.update({"learning_rate_start": self.learning_rate_start,
                       "learning_rate_stop": self.learning_rate_stop, "epo": self.epo, "epo_min":self.epo_min})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LinearWarmupExponentialDecay')
class LinearWarmupExponentialDecay(tf.optimizers.schedules.LearningRateSchedule):
    """This schedule combines a linear warmup with an exponential decay."""

    def __init__(self, learning_rate, warmup_steps, decay_steps, decay_rate):
        super().__init__()
        self._input_config_settings = {"learning_rate": float(learning_rate), "warmup_steps": int(warmup_steps),
                                       "decay_steps": int(decay_steps), "decay_rate": int(decay_rate)}
        self.warmup = tf.optimizers.schedules.PolynomialDecay(
            1 / warmup_steps, warmup_steps, end_learning_rate=1)
        self.decay = tf.optimizers.schedules.ExponentialDecay(
            learning_rate, decay_steps, decay_rate)

    def __call__(self, step):
        return self.warmup(step) * self.decay(step)

    def get_config(self):
        """Get config."""
        config = {}
        config.update(self._input_config_settings)
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LearningRateLoggingCallback')
class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
    """Callback logging the learning rate."""

    def __init__(self, verbose=0):
        super(LearningRateLoggingCallback, self).__init__()
        self.verbose=verbose

    def on_epoch_end(self, epoch, logs=None):
        """Read out the learning rate on epoch end.

        Args:
            epoch (int, float): Number of current epoch ended.
            logs (dict): Dictionary of the logs.

        Returns:
            None.
        """
        lr = self.model.optimizer.lr
        tf.summary.scalar('learning rate', data=lr, step=epoch)
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (epoch + 1, lr))

    def get_config(self):
        """Get config."""
        config = {"verbose": self.verbose}
        return config

import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LearningRateLoggingCallback')
class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
    """Callback logging the learning rate. This does not work universally yet.
    Will be improved in the future."""

    def __init__(self, verbose: int = 0):
        super(LearningRateLoggingCallback, self).__init__()
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        """Read out the learning rate on epoch end.

        Args:
            epoch (int): Number of current epoch ended.
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

# import numpy as np
import tensorflow as tf
ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='LearningRateLoggingCallback')
class LearningRateLoggingCallback(ks.callbacks.Callback):
    """Callback logging the learning rate."""

    def __init__(self, verbose: int = 1):
        """Initialize class.

        Args:
            verbose (int): Verbosity. Default is 1.
        """
        super(LearningRateLoggingCallback, self).__init__()
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        """Read out the learning rate on epoch end.

        Args:
            epoch (int): Number of current epoch.
            logs (dict): Dictionary of the logs.

        Returns:
            None.
        """
        lr = self.model.optimizer.lr
        tf.summary.scalar('learning rate', data=lr, step=epoch)
        logs = logs or {}
        logs['lr'] = ks.backend.get_value(self.model.optimizer.lr)
        if self.verbose > 0:
            print("\nEpoch %05d: Finished epoch with learning rate: %s.\n" % (epoch + 1, logs['lr']))

    def get_config(self):
        """Get config for this class."""
        config = {"verbose": self.verbose}
        return config

    @classmethod
    def from_config(cls, config):
        """Make class instance from config."""
        return cls(**config)

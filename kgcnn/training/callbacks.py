import keras as ks
import keras.callbacks


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
        lr = self.model.optimizer.learning_rate
        logs = logs or {}
        logs['lr'] = float(ks.backend.convert_to_numpy(self.model.optimizer.learning_rate))
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

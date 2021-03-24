import tensorflow as tf
import tensorflow.keras as ks


class ScaledMeanAbsoluteError(tf.keras.metrics.MeanAbsoluteError):

    def __init__(self, scaling_shape=(), name='mean_absolute_error', **kwargs):
        super(ScaledMeanAbsoluteError, self).__init__(name=name, **kwargs)
        self.scale = self.add_weight(shape=scaling_shape, initializer=tf.keras.initializers.Ones(), name='scale_mae',
                                     dtype=tf.keras.backend.floatx())
        self.scaling_shape = scaling_shape

    def reset_states(self):
        # Super variables
        ks.backend.set_value(self.total, 0)
        ks.backend.set_value(self.count, 0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = self.scale * y_true
        y_pred = self.scale * y_pred
        return super(ScaledMeanAbsoluteError, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        """Returns the serializable config of the metric."""
        mae_conf = super(ScaledMeanAbsoluteError, self).get_config()
        mae_conf.update({"scaling_shape": self.scaling_shape})
        return mae_conf

    def set_scale(self, scale):
        ks.backend.set_value(self.scale, scale)

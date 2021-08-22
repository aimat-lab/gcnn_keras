import tensorflow as tf
import tensorflow.keras as ks


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ScaledMeanAbsoluteError')
class ScaledMeanAbsoluteError(tf.keras.metrics.MeanAbsoluteError):
    """Metric for a scaled mean absolute error (MAE), which can undo a pre-scaling of the targets. Only intended as
    metric this allows to log the MAE with correct units or absolute values during fit."""

    def __init__(self, scaling_shape=(), name='mean_absolute_error', **kwargs):
        super(ScaledMeanAbsoluteError, self).__init__(name=name, **kwargs)
        self.scale = self.add_weight(shape=scaling_shape,
                                     initializer=tf.keras.initializers.Ones(), name='kgcnn_scale_mae',
                                     dtype=tf.keras.backend.floatx())
        self.scaling_shape = scaling_shape

    def reset_state(self):
        ks.backend.batch_set_value([(v, 0) for v in self.variables if 'kgcnn_scale_mae' not in v.name])
        # Or set them explicitly.
        # ks.backend.set_value(self.total, 0)
        # ks.backend.set_value(self.count, 0)

    def reset_states(self):
        ks.backend.batch_set_value([(v, 0) for v in self.variables if 'kgcnn_scale_mae' not in v.name])
        # Or set them explicitly.
        # ks.backend.set_value(self.total, 0)
        # ks.backend.set_value(self.count, 0)

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
        """Set the scale from numpy array. Usually used with broadcasting."""
        ks.backend.set_value(self.scale, scale)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ScaledRootMeanSquaredError')
class ScaledRootMeanSquaredError(tf.keras.metrics.RootMeanSquaredError):
    """Metric for a scaled root mean squared error (RMSE), which can undo a pre-scaling of the targets.
    Only intended as metric this allows to log the MAE with correct units or absolute values during fit."""

    def __init__(self, scaling_shape=(), name='root_mean_squared_error', **kwargs):
        super(ScaledRootMeanSquaredError, self).__init__(name=name, **kwargs)
        self.scale = self.add_weight(shape=scaling_shape,
                                     initializer=tf.keras.initializers.Ones(), name='kgcnn_scale_rmse',
                                     dtype=tf.keras.backend.floatx())
        self.scaling_shape = scaling_shape

    def reset_state(self):
        ks.backend.batch_set_value([(v, 0) for v in self.variables if 'kgcnn_scale_rmse' not in v.name])
        # Or set them explicitly.
        # ks.backend.set_value(self.total, 0)
        # ks.backend.set_value(self.count, 0)

    def reset_states(self):
        ks.backend.batch_set_value([(v, 0) for v in self.variables if 'kgcnn_scale_rmse' not in v.name])
        # Or set them explicitly.
        # ks.backend.set_value(self.total, 0)
        # ks.backend.set_value(self.count, 0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = self.scale * y_true
        y_pred = self.scale * y_pred
        return super(ScaledRootMeanSquaredError, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        """Returns the serializable config of the metric."""
        mae_conf = super(ScaledRootMeanSquaredError, self).get_config()
        mae_conf.update({"scaling_shape": self.scaling_shape})
        return mae_conf

    def set_scale(self, scale):
        """Set the scale from numpy array. Usually used with broadcasting."""
        ks.backend.set_value(self.scale, scale)

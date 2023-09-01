import keras_core as ks
import keras_core.metrics
from keras_core import ops


class ScaledMeanAbsoluteError(ks.metrics.MeanAbsoluteError):
    """Metric for a scaled mean absolute error (MAE), which can undo a pre-scaling of the targets. Only intended as
    metric this allows to info the MAE with correct units or absolute values during fit."""

    def __init__(self, scaling_shape=(), name='mean_absolute_error', dtype_scale: str = None, **kwargs):
        super(ScaledMeanAbsoluteError, self).__init__(name=name, **kwargs)
        self.scale = self.add_variable(
            shape=scaling_shape,
            initializer=ks.initializers.Ones(),
            name='kgcnn_scale_mae',
            dtype=dtype_scale
        )
        self.scaling_shape = scaling_shape
        self.dtype_scale = dtype_scale

    def reset_state(self):
        for v in self.variables:
            if 'kgcnn_scale_mae' not in v.name:
                v.assign(ops.zeros(v.shape, dtype=v.dtype))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = self.scale * y_true
        y_pred = self.scale * y_pred
        return super(ScaledMeanAbsoluteError, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        """Returns the serializable config of the metric."""
        conf = super(ScaledMeanAbsoluteError, self).get_config()
        conf.update({"scaling_shape": self.scaling_shape, "dtype_scale": self.dtype_scale})
        return conf

    def set_scale(self, scale):
        """Set the scale from numpy array. Usually used with broadcasting."""
        self.scale.assign(ops.cast(scale, dtype=scale.dtype))


class ScaledRootMeanSquaredError(ks.metrics.RootMeanSquaredError):
    """Metric for a scaled root mean squared error (RMSE), which can undo a pre-scaling of the targets.
    Only intended as metric this allows to info the MAE with correct units or absolute values during fit."""

    def __init__(self, scaling_shape=(), name='root_mean_squared_error', dtype_scale: str = None, **kwargs):
        super(ScaledRootMeanSquaredError, self).__init__(name=name, **kwargs)
        self.scale = self.add_variable(
            shape=scaling_shape,
            initializer=ks.initializers.Ones(),
            name='kgcnn_scale_rmse',
            dtype=dtype_scale
        )
        self.scaling_shape = scaling_shape
        self.dtype_scale = dtype_scale

    def reset_state(self):
        for v in self.variables:
            if 'kgcnn_scale_rmse' not in v.name:
                v.assign(ops.zeros(v.shape, dtype=v.dtype))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = self.scale * y_true
        y_pred = self.scale * y_pred
        return super(ScaledRootMeanSquaredError, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        """Returns the serializable config of the metric."""
        conf = super(ScaledRootMeanSquaredError, self).get_config()
        conf.update({"scaling_shape": self.scaling_shape, "dtype_scale": self.dtype_scale})
        return conf

    def set_scale(self, scale):
        """Set the scale from numpy array. Usually used with broadcasting."""
        self.scale.assign(ops.cast(scale, dtype=scale.dtype))

import keras_core as ks
import keras_core.metrics
import numpy as np
from keras_core import ops
import keras_core.saving


@ks.saving.register_keras_serializable(package='kgcnn', name='ScaledMeanAbsoluteError')
class ScaledMeanAbsoluteError(ks.metrics.MeanAbsoluteError):
    """Metric for a scaled mean absolute error (MAE), which can undo a pre-scaling of the targets. Only intended as
    metric this allows to info the MAE with correct units or absolute values during fit."""

    def __init__(self, scaling_shape=(), name='mean_absolute_error', dtype_scale: str = None, **kwargs):
        super(ScaledMeanAbsoluteError, self).__init__(name=name, **kwargs)
        self.scaling_shape = scaling_shape
        self.dtype_scale = dtype_scale
        self.scale = self.add_variable(
            shape=scaling_shape,
            initializer=ks.initializers.Ones(),
            name='kgcnn_scale_mae',
            dtype=self.dtype_scale if self.dtype_scale is not None else self.dtype
        )

    def reset_state(self):
        for v in self.variables:
            if 'kgcnn_scale_mae' not in v.name:
                v.assign(ops.zeros(v.shape, dtype=v.dtype))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = self.scale * ops.cast(y_true, dtype=self.scale.dtype)
        y_pred = self.scale * ops.cast(y_pred, dtype=self.scale.dtype)
        return super(ScaledMeanAbsoluteError, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        """Returns the serializable config of the metric."""
        conf = super(ScaledMeanAbsoluteError, self).get_config()
        conf.update({"scaling_shape": self.scaling_shape, "dtype_scale": self.dtype_scale})
        return conf

    def set_scale(self, scale):
        """Set the scale from numpy array. Usually used with broadcasting."""
        self.scale.assign(ops.cast(scale, dtype=scale.dtype))


@ks.saving.register_keras_serializable(package='kgcnn', name='ScaledRootMeanSquaredError')
class ScaledRootMeanSquaredError(ks.metrics.RootMeanSquaredError):
    """Metric for a scaled root mean squared error (RMSE), which can undo a pre-scaling of the targets.
    Only intended as metric this allows to info the MAE with correct units or absolute values during fit."""

    def __init__(self, scaling_shape=(), name='root_mean_squared_error', dtype_scale: str = None, **kwargs):
        super(ScaledRootMeanSquaredError, self).__init__(name=name, **kwargs)
        self.scaling_shape = scaling_shape
        self.dtype_scale = dtype_scale
        self.scale = self.add_variable(
            shape=scaling_shape,
            initializer=ks.initializers.Ones(),
            name='kgcnn_scale_rmse',
            dtype=self.dtype_scale if self.dtype_scale is not None else self.dtype
        )

    def reset_state(self):
        for v in self.variables:
            if 'kgcnn_scale_rmse' not in v.name:
                v.assign(ops.zeros(v.shape, dtype=v.dtype))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = self.scale * ops.cast(y_true, dtype=self.scale.dtype)
        y_pred = self.scale * ops.cast(y_pred, dtype=self.scale.dtype)
        return super(ScaledRootMeanSquaredError, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        """Returns the serializable config of the metric."""
        conf = super(ScaledRootMeanSquaredError, self).get_config()
        conf.update({"scaling_shape": self.scaling_shape, "dtype_scale": self.dtype_scale})
        return conf

    def set_scale(self, scale):
        """Set the scale from numpy array. Usually used with broadcasting."""
        self.scale.assign(ops.cast(scale, dtype=scale.dtype))


@ks.saving.register_keras_serializable(package='kgcnn', name='ScaledMeanAbsoluteError')
class ScaledForceMeanAbsoluteError(ks.metrics.MeanMetricWrapper):
    """Metric for a scaled mean absolute error (MAE), which can undo a pre-scaling of the targets. Only intended as
    metric this allows to info the MAE with correct units or absolute values during fit."""

    def __init__(self, scaling_shape=(), name='force_mean_absolute_error', dtype_scale: str = None,
                 squeeze_states: bool = True, **kwargs):
        super(ScaledForceMeanAbsoluteError, self).__init__(fn=self.fn_force_mae, name=name, **kwargs)
        self.scaling_shape = scaling_shape
        self.dtype_scale = dtype_scale
        self.squeeze_states = squeeze_states

        if scaling_shape[-1] == 1 and squeeze_states and len(scaling_shape) > 1:
            scaling_shape = scaling_shape[:-1]
        scaling_shape = tuple(list(scaling_shape[:1]) + [1, 1] + list(scaling_shape[1:]))

        self.scale = self.add_variable(
            shape=scaling_shape,
            initializer=ks.initializers.Ones(),
            name='kgcnn_scale_mae',
            dtype=self.dtype_scale if self.dtype_scale is not None else self.dtype
        )

    def reset_state(self):
        for v in self.variables:
            if 'kgcnn_scale_mae' not in v.name:
                v.assign(ops.zeros(v.shape, dtype=v.dtype))

    def fn_force_mae(self, y_true, y_pred):
        check_nonzero = ops.logical_not(
            ops.all(ops.isclose(y_true, ops.convert_to_tensor(0., dtype=y_true.dtype)), axis=2))
        row_count = ops.cast(ops.sum(check_nonzero, axis=1), dtype=y_pred.dtype)

        y_true = self.scale * ops.cast(y_true, dtype=self.scale.dtype)
        y_pred = self.scale * ops.cast(y_pred, dtype=self.scale.dtype)
        diff = ops.abs(y_true-y_pred)

        if not self.squeeze_states:
            diff = ops.mean(diff, axis=3)

        out = ops.sum(ops.mean(diff, axis=2), axis=1)/row_count
        return out

    def get_config(self):
        """Returns the serializable config of the metric."""
        conf = super(ScaledForceMeanAbsoluteError, self).get_config()
        conf.update({"scaling_shape": self.scaling_shape, "dtype_scale": self.dtype_scale})
        return conf

    def set_scale(self, scale):
        """Set the scale from numpy array. Usually used with broadcasting."""
        scaling_shape = scale.shape
        if scaling_shape[-1] == 1 and self.squeeze_states and len(scaling_shape) > 1:
            scale = np.squeeze(scale, axis=-1)
        scale = np.expand_dims(np.expand_dims(scale, axis=1), axis=2)
        self.scale.assign(ops.cast(scale, dtype=scale.dtype))

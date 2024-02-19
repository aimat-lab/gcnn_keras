import keras as ks
import keras.metrics
import numpy as np
from keras import ops
import keras.saving
from kgcnn.ops.core import decompose_ragged_tensor


@ks.saving.register_keras_serializable(package='kgcnn', name='ScaledMeanAbsoluteError')
class ScaledMeanAbsoluteError(ks.metrics.MeanAbsoluteError):
    """Metric for a scaled mean absolute error (MAE), which can undo a pre-scaling of the targets. Only intended as
    metric this allows to info the MAE with correct units or absolute values during fit."""

    def __init__(self, scaling_shape=(), name='mean_absolute_error', dtype_scale: str = None, ragged: bool = False,
                 **kwargs):
        super(ScaledMeanAbsoluteError, self).__init__(name=name, **kwargs)
        self.scaling_shape = scaling_shape
        self._is_ragged = ragged
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
        if self._is_ragged:
            y_true = decompose_ragged_tensor(y_true)[0]
            y_pred = decompose_ragged_tensor(y_pred)[0]
        y_true = self.scale * ops.cast(y_true, dtype=self.scale.dtype)
        y_pred = self.scale * ops.cast(y_pred, dtype=self.scale.dtype)
        return super(ScaledMeanAbsoluteError, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        """Returns the serializable config of the metric."""
        conf = super(ScaledMeanAbsoluteError, self).get_config()
        conf.update({"scaling_shape": self.scaling_shape, "dtype_scale": self.dtype_scale,
                     "ragged": self._is_ragged})
        return conf

    def set_scale(self, scale):
        """Set the scale from numpy array. Usually used with broadcasting."""
        self.scale.assign(ops.cast(scale, dtype=scale.dtype))


@ks.saving.register_keras_serializable(package='kgcnn', name='ScaledRootMeanSquaredError')
class ScaledRootMeanSquaredError(ks.metrics.RootMeanSquaredError):
    """Metric for a scaled root mean squared error (RMSE), which can undo a pre-scaling of the targets.
    Only intended as metric this allows to info the MAE with correct units or absolute values during fit."""

    def __init__(self, scaling_shape=(), name='root_mean_squared_error', dtype_scale: str = None, ragged: bool = False,
                 **kwargs):
        super(ScaledRootMeanSquaredError, self).__init__(name=name, **kwargs)
        self.scaling_shape = scaling_shape
        self.dtype_scale = dtype_scale
        self._is_ragged = ragged
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
        if self._is_ragged:
            y_true = decompose_ragged_tensor(y_true)[0]
            y_pred = decompose_ragged_tensor(y_pred)[0]
        y_true = self.scale * ops.cast(y_true, dtype=self.scale.dtype)
        y_pred = self.scale * ops.cast(y_pred, dtype=self.scale.dtype)
        return super(ScaledRootMeanSquaredError, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        """Returns the serializable config of the metric."""
        conf = super(ScaledRootMeanSquaredError, self).get_config()
        conf.update({"scaling_shape": self.scaling_shape, "dtype_scale": self.dtype_scale, "ragged": self._is_ragged})
        return conf

    def set_scale(self, scale):
        """Set the scale from numpy array. Usually used with broadcasting."""
        self.scale.assign(ops.cast(scale, dtype=scale.dtype))


@ks.saving.register_keras_serializable(package='kgcnn', name='ScaledMeanAbsoluteError')
class ScaledForceMeanAbsoluteError(ks.metrics.MeanMetricWrapper):
    """Metric for a scaled mean absolute error (MAE), which can undo a pre-scaling of the targets. Only intended as
    metric this allows to info the MAE with correct units or absolute values during fit."""

    def __init__(self, scaling_shape=(1, 1), name='force_mean_absolute_error', dtype_scale: str = None,
                 squeeze_states: bool = True, find_padded_atoms: bool = True, **kwargs):
        super(ScaledForceMeanAbsoluteError, self).__init__(fn=self.fn_force_mae, name=name, **kwargs)
        self.scaling_shape = scaling_shape
        self.dtype_scale = dtype_scale
        self.squeeze_states = squeeze_states
        self.find_padded_atoms = find_padded_atoms

        if scaling_shape[-1] == 1 and squeeze_states and len(scaling_shape) > 1:
            scaling_shape = scaling_shape[:-1]
        scaling_shape = tuple(list(scaling_shape[:1]) + [1, 1] + list(scaling_shape[1:]))

        self.scale = self.add_variable(
            shape=scaling_shape,
            initializer=ks.initializers.Ones(),
            name='kgcnn_scale_mae',
            dtype=self.dtype_scale if self.dtype_scale is not None else self.dtype
        )

    def fn_force_mae(self, y_true, y_pred):
        # (batch, N, 3)
        if self.find_padded_atoms:
            check_nonzero = ops.cast(ops.logical_not(
                ops.all(ops.isclose(y_true, ops.convert_to_tensor(0., dtype=y_true.dtype)), axis=2)), dtype="int32")
            row_count = ops.sum(check_nonzero, axis=1)
            row_count = ops.where(row_count < 1, 1, row_count)
            norm = 1/ops.cast(row_count, dtype=self.scale.dtype)
        else:
            norm = 1/ops.shape(y_true)[1]

        y_true = self.scale * ops.cast(y_true, dtype=self.scale.dtype)
        y_pred = self.scale * ops.cast(y_pred, dtype=self.scale.dtype)

        diff = ops.abs(y_true-y_pred)
        out = ops.sum(ops.mean(diff, axis=2), axis=1)*norm
        if not self.squeeze_states:
            out = ops.mean(out, axis=-1)
        return out

    def reset_state(self):
        for v in self.variables:
            if 'kgcnn_scale_mae' not in v.name:
                v.assign(ops.zeros(v.shape, dtype=v.dtype))

    def get_config(self):
        """Returns the serializable config of the metric."""
        # May not manage to deserialize `fn_force_mae`, set conf directly.
        # conf = super(ScaledForceMeanAbsoluteError, self).get_config()
        conf = {"name": self.name, "dtype": self.dtype}
        conf.update({"scaling_shape": self.scaling_shape, "dtype_scale": self.dtype_scale,
                     "find_padded_atoms": self.find_padded_atoms, "squeeze_states": self.squeeze_states})
        return conf

    def set_scale(self, scale):
        """Set the scale from numpy array. Usually used with broadcasting."""
        scaling_shape = scale.shape
        if scaling_shape[-1] == 1 and self.squeeze_states and len(scaling_shape) > 1:
            scale = np.squeeze(scale, axis=-1)
        scale = np.expand_dims(np.expand_dims(scale, axis=1), axis=2)
        self.scale.assign(ops.cast(scale, dtype=scale.dtype))


@ks.saving.register_keras_serializable(package='kgcnn', name='BinaryAccuracyNoNaN')
class BinaryAccuracyNoNaN(ks.metrics.MeanMetricWrapper):

    def __init__(self, name="binary_accuracy_no_nan",  dtype=None, threshold=0.5, **kwargs):
        if threshold is not None and (threshold <= 0 or threshold >= 1):
            raise ValueError(
                "Invalid value for argument `threshold`. "
                "Expected a value in interval (0, 1). "
                f"Received: threshold={threshold}"
            )
        super().__init__(
            fn=self._binary_accuracy_no_nan, name=name, dtype=dtype, threshold=threshold, **kwargs
        )
        self.threshold = threshold

    @staticmethod
    def _binary_accuracy_no_nan(y_true, y_pred, threshold=0.5):
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        is_not_nan = ops.cast(ops.logical_not(ops.isnan(y_true)), y_true.dtype)
        threshold = ops.cast(threshold, y_pred.dtype)
        y_pred = ops.cast(y_pred > threshold, y_true.dtype)
        counts = ops.sum(ops.cast(
            ops.equal(y_true, y_pred), dtype=ks.backend.floatx()), axis=-1)
        norm = ops.sum(ops.cast(is_not_nan, dtype=ks.backend.floatx()), axis=-1)
        return counts/norm

    def get_config(self):
        config = {"name": self.name, "dtype": self.dtype, "threshold": self.threshold}
        return config


@ks.saving.register_keras_serializable(package='kgcnn', name='AUCNoNaN')
class AUCNoNaN(ks.metrics.AUC):

    def __init__(self, name="AUC_no_nan", **kwargs):
        super(AUCNoNaN, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        is_not_nan = ops.cast(ops.logical_not(ops.isnan(y_true)), y_true.dtype)
        if sample_weight is not None:
            sample_weight *= is_not_nan
        else:
            sample_weight = is_not_nan
        return super(AUCNoNaN, self).update_state(y_true, y_pred, sample_weight=sample_weight)


@ks.saving.register_keras_serializable(package='kgcnn', name='BalancedBinaryAccuracyNoNaN')
class BalancedBinaryAccuracyNoNaN(ks.metrics.SensitivityAtSpecificity):

    def __init__(self, name="balanced_binary_accuracy_no_nan", class_id=None, num_thresholds=1,
                 specificity=0.5, **kwargs):
        super(BalancedBinaryAccuracyNoNaN, self).__init__(name=name, class_id=class_id, num_thresholds=num_thresholds,
                                                          specificity=specificity, **kwargs)
        self._thresholds_distributed_evenly = False

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the state of the metric.

        Args:
            y_true: Ground truth label values. shape = `[batch_size, d0, .. dN-1]`
                or shape = `[batch_size, d0, .. dN-1, 1]` .
            y_pred: The predicted probability values. shape = `[batch_size, d0, .. dN]` .
            sample_weight: Optional sample_weight acts as a coefficient for the metric.
        """
        is_not_nan = ops.cast(ops.logical_not(ops.isnan(y_true)), y_true.dtype)
        if sample_weight is not None:
            sample_weight *= is_not_nan
        else:
            sample_weight = is_not_nan

        return super(BalancedBinaryAccuracyNoNaN, self).update_state(
            y_true=y_true, y_pred=y_pred,
            sample_weight=sample_weight
        )

    def result(self):
        sensitivities = ops.divide(
            self.true_positives,
            self.true_positives + self.false_negatives + ks.config.epsilon(),
        )
        specificities = ops.divide(
            self.true_negatives,
            self.true_negatives + self.false_positives + ks.config.epsilon(),
        )
        result = (sensitivities + specificities)/2
        return result

    def get_config(self):
        config = super(BalancedBinaryAccuracyNoNaN, self).get_config()
        return config

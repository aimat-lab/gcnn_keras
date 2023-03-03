import numpy as np
import tensorflow as tf
ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='ScaledMeanAbsoluteError')
class ScaledMeanAbsoluteError(ks.metrics.MeanAbsoluteError):
    """Metric for a scaled mean absolute error (MAE), which can undo a pre-scaling of the targets. Only intended as
    metric this allows to info the MAE with correct units or absolute values during fit."""

    def __init__(self, scaling_shape=(), name='mean_absolute_error', **kwargs):
        super(ScaledMeanAbsoluteError, self).__init__(name=name, **kwargs)
        self.scale = self.add_weight(shape=scaling_shape,
                                     initializer=ks.initializers.Ones(),
                                     name='kgcnn_scale_mae',
                                     dtype=ks.backend.floatx(),
                                     synchronization=tf.VariableSynchronization.AUTO,
                                     aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                                     )
        self.scaling_shape = scaling_shape

    def reset_state(self):
        ks.backend.batch_set_value([(v, 0) for v in self.variables if 'kgcnn_scale_mae' not in v.name])

    def reset_states(self):
        ks.backend.batch_set_value([(v, 0) for v in self.variables if 'kgcnn_scale_mae' not in v.name])

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

    def merge_state(self, metrics):
        assign_add_ops = []
        for metric in metrics:
            if len(self.weights) != len(metric.weights):
                raise ValueError(f'Metric {metric} is not compatible with {self}')
            for weight, weight_to_add in zip(self.weights, metric.weights):
                if 'kgcnn_scale_mae' not in weight.name:
                    assign_add_ops.append(weight.assign_add(weight_to_add))
        return assign_add_ops


@ks.utils.register_keras_serializable(package='kgcnn', name='RaggedScaledMeanAbsoluteError')
class RaggedScaledMeanAbsoluteError(ks.metrics.MeanAbsoluteError):
    """Metric for a scaled mean absolute error (MAE), which can undo a pre-scaling of the targets. Only intended as
    metric this allows to info the MAE with correct units or absolute values during fit."""

    def __init__(self, scaling_shape=(), name='mean_absolute_error', **kwargs):
        super(RaggedScaledMeanAbsoluteError, self).__init__(name=name, **kwargs)
        self.scale = self.add_weight(shape=scaling_shape,
                                     initializer=ks.initializers.Ones(),
                                     name='kgcnn_scale_mae',
                                     dtype=ks.backend.floatx(),
                                     synchronization=tf.VariableSynchronization.AUTO,
                                     aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                                     )
        self.scaling_shape = scaling_shape

    def reset_state(self):
        ks.backend.batch_set_value([(v, 0) for v in self.variables if 'kgcnn_scale_mae' not in v.name])

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = [tf.ragged.map_flat_values(lambda x: x * self.scale, y) for y in [y_true, y_pred]]
        # return super(RaggedScaledMeanAbsoluteError, self).update_state(
        #   y_true.flat_values, y_pred.flat_values, sample_weight=sample_weight)
        return super(RaggedScaledMeanAbsoluteError, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        """Returns the serializable config of the metric."""
        mae_conf = super(RaggedScaledMeanAbsoluteError, self).get_config()
        mae_conf.update({"scaling_shape": self.scaling_shape})
        return mae_conf

    def set_scale(self, scale):
        """Set the scale from numpy array. Usually used with broadcasting."""
        ks.backend.set_value(self.scale, scale)

    def merge_state(self, metrics):
        assign_add_ops = []
        for metric in metrics:
            if len(self.weights) != len(metric.weights):
                raise ValueError(f'Metric {metric} is not compatible with {self}')
            for weight, weight_to_add in zip(self.weights, metric.weights):
                if 'kgcnn_scale_mae' not in weight.name:
                    assign_add_ops.append(weight.assign_add(weight_to_add))
        return assign_add_ops


@ks.utils.register_keras_serializable(package='kgcnn', name='ScaledRootMeanSquaredError')
class ScaledRootMeanSquaredError(ks.metrics.RootMeanSquaredError):
    """Metric for a scaled root mean squared error (RMSE), which can undo a pre-scaling of the targets.
    Only intended as metric this allows to info the MAE with correct units or absolute values during fit."""

    def __init__(self, scaling_shape=(), name='root_mean_squared_error', **kwargs):
        super(ScaledRootMeanSquaredError, self).__init__(name=name, **kwargs)
        self.scale = self.add_weight(shape=scaling_shape,
                                     initializer=ks.initializers.Ones(),
                                     name='kgcnn_scale_rmse',
                                     dtype=ks.backend.floatx(),
                                     synchronization=tf.VariableSynchronization.AUTO,
                                     aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                                     )
        self.scaling_shape = scaling_shape

    def reset_state(self):
        ks.backend.batch_set_value([(v, 0) for v in self.variables if 'kgcnn_scale_rmse' not in v.name])

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

    def merge_state(self, metrics):
        assign_add_ops = []
        for metric in metrics:
            if len(self.weights) != len(metric.weights):
                raise ValueError(f'Metric {metric} is not compatible with {self}')
            for weight, weight_to_add in zip(self.weights, metric.weights):
                if 'kgcnn_scale_rmse' not in weight.name:
                    assign_add_ops.append(weight.assign_add(weight_to_add))
        return assign_add_ops


@ks.utils.register_keras_serializable(package='kgcnn', name='BinaryAccuracyNoNaN')
class BinaryAccuracyNoNaN(ks.metrics.BinaryAccuracy):

    def __init__(self, name="binary_accuracy_no_nan", **kwargs):
        super(BinaryAccuracyNoNaN, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        is_not_nan = tf.cast(tf.logical_not(tf.math.is_nan(y_true)), y_pred.dtype)
        if sample_weight is not None:
            sample_weight *= is_not_nan
        else:
            sample_weight = is_not_nan
        return super(BinaryAccuracyNoNaN, self).update_state(y_true, y_pred, sample_weight=sample_weight)


@ks.utils.register_keras_serializable(package='kgcnn', name='AUCNoNaN')
class AUCNoNaN(ks.metrics.AUC):

    def __init__(self, name="AUC_no_nan", **kwargs):
        super(AUCNoNaN, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        is_not_nan = tf.cast(tf.logical_not(tf.math.is_nan(y_true)), y_pred.dtype)
        if sample_weight is not None:
            sample_weight *= is_not_nan
        else:
            sample_weight = is_not_nan
        return super(AUCNoNaN, self).update_state(y_true, y_pred, sample_weight=sample_weight)


@ks.utils.register_keras_serializable(package='kgcnn', name='BalancedBinaryAccuracyNoNaN')
class BalancedBinaryAccuracyNoNaN(ks.metrics.Metric):

    def __init__(self, name="balanced_binary_accuracy_no_nan", thresholds=None, **kwargs):
        super(BalancedBinaryAccuracyNoNaN, self).__init__(name=name, **kwargs)
        self.thresholds = thresholds
        self.confusion_metrics = [
            tf.keras.metrics.TruePositives(thresholds=thresholds, **kwargs),
            tf.keras.metrics.FalsePositives(thresholds=thresholds, **kwargs),
            tf.keras.metrics.TrueNegatives(thresholds=thresholds, **kwargs),
            tf.keras.metrics.FalseNegatives(thresholds=thresholds, **kwargs)
        ]

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the state of the metric.

        Args:
            y_true: Ground truth label values. shape = `[batch_size, d0, .. dN-1]`
                or shape = `[batch_size, d0, .. dN-1, 1]` .
            y_pred: The predicted probability values. shape = `[batch_size, d0, .. dN]` .
            sample_weight: Optional sample_weight acts as a coefficient for the metric.

        Returns:
            update ops.
        """

        is_not_nan = tf.cast(tf.logical_not(tf.math.is_nan(y_true)), y_pred.dtype)
        if sample_weight is not None:
            sample_weight *= is_not_nan
        else:
            sample_weight = is_not_nan

        update_ops = [m.update_state(y_true, y_pred, sample_weight=sample_weight) for m in self.confusion_metrics]
        return tf.group(update_ops)

    def result(self):
        true_positives, false_positives, true_negatives, false_negatives = [
            m.result() for m in self.confusion_metrics
        ]
        result = (tf.math.divide_no_nan(true_positives, true_positives + false_negatives) +
                  tf.math.divide_no_nan(true_negatives, true_negatives + false_positives)) / 2
        return result

    def reset_state(self):
        for m in self.confusion_metrics:
            m.reset_state()

    def get_config(self):
        config = super(BalancedBinaryAccuracyNoNaN, self).get_config()
        config.update({
            "thresholds": self.thresholds,
        })
        return config

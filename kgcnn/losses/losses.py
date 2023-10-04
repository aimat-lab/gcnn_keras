import keras_core as ks
from keras_core import ops
from keras_core.losses import Loss
from keras_core.losses import mean_absolute_error
import keras_core.saving


@ks.saving.register_keras_serializable(package='kgcnn', name='MeanAbsoluteError')
class MeanAbsoluteError(Loss):

    def __init__(self, reduction="sum_over_batch_size", name="mean_absolute_error", dtype=None):
        super(MeanAbsoluteError, self).__init__(reduction=reduction, name=name, dtype=dtype)

    def call(self, y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    def get_config(self):
        config = super(MeanAbsoluteError, self).get_config()
        return config


@ks.saving.register_keras_serializable(package='kgcnn', name='ForceMeanAbsoluteError')
class ForceMeanAbsoluteError(Loss):

    def __init__(self, reduction="sum_over_batch_size", name="force_mean_absolute_error",
                 squeeze_states: bool = True, dtype=None):
        super(ForceMeanAbsoluteError, self).__init__(reduction=reduction, name=name, dtype=dtype)
        self.squeeze_states = squeeze_states

    def call(self, y_true, y_pred):
        # Shape: (batch, N, 3, S)
        check_nonzero = ops.logical_not(
            ops.all(ops.isclose(y_true, ops.convert_to_tensor(0., dtype=y_true.dtype)), axis=2))
        row_count = ops.cast(ops.sum(check_nonzero, axis=1), dtype=y_true.dtype)
        norm = 1/row_count
        norm = ops.where(ops.isnan(norm), 0., norm)

        diff = ops.abs(y_true-y_pred)
        out = ops.sum(ops.mean(diff, axis=2), axis=1)*norm
        if not self.squeeze_states:
            out = ops.mean(out, axis=-1)
        return out

    def get_config(self):
        config = super(ForceMeanAbsoluteError, self).get_config()
        return config
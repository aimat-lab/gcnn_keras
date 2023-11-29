import keras as ks
from keras import ops
from keras.losses import Loss
from keras.losses import mean_absolute_error
import keras.saving


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
                 squeeze_states: bool = True, find_padded_atoms: bool = True, dtype=None):
        super(ForceMeanAbsoluteError, self).__init__(reduction=reduction, name=name, dtype=dtype)
        self.squeeze_states = squeeze_states
        self.find_padded_atoms = find_padded_atoms

    def call(self, y_true, y_pred):
        # Shape: (batch, N, 3, S)
        if self.find_padded_atoms:
            check_nonzero = ops.cast(ops.logical_not(
                ops.all(ops.isclose(y_true, ops.convert_to_tensor(0., dtype=y_true.dtype)), axis=2)), dtype="int32")
            row_count = ops.sum(check_nonzero, axis=1)
            row_count = ops.where(row_count < 1, 1, row_count)
            norm = 1/ops.cast(row_count, dtype=y_true.dtype)
        else:
            norm = 1/ops.shape(y_true)[1]

        diff = ops.abs(y_true-y_pred)
        out = ops.sum(ops.mean(diff, axis=2), axis=1)*norm
        if not self.squeeze_states:
            out = ops.mean(out, axis=-1)
        return out

    def get_config(self):
        config = super(ForceMeanAbsoluteError, self).get_config()
        config.update({"find_padded_atoms": self.find_padded_atoms, "squeeze_states": self.squeeze_states})
        return config


@ks.saving.register_keras_serializable(package='kgcnn', name='BinaryCrossentropyNoNaN')
class BinaryCrossentropyNoNaN(ks.losses.BinaryCrossentropy):

    def __init__(self, *args, **kwargs):
        super(BinaryCrossentropyNoNaN, self).__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        is_nan = ops.isnan(y_true)
        y_pred = ops.where(is_nan, ops.zeros_like(y_pred), y_pred)
        y_true = ops.where(is_nan, ops.zeros_like(y_true), y_true)
        return super(BinaryCrossentropyNoNaN, self).call(y_true, y_pred)

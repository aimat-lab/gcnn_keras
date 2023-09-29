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

    def __init__(self, reduction="sum_over_batch_size", name="mean_absolute_error", dtype=None):
        super(ForceMeanAbsoluteError, self).__init__(reduction=reduction, name=name, dtype=dtype)

    def call(self, y_true, y_pred):
        check_nonzero = ops.all(ops.isclose(y_true, 0.), axis=2)
        row_count = ops.sum(check_nonzero, axis=1)
        return ops.sum(ops.abs(y_true-y_pred), axis=(1, 2))/row_count

    def get_config(self):
        config = super(ForceMeanAbsoluteError, self).get_config()
        return config
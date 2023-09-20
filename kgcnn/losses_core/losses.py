import keras_core as ks
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
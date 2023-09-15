import tensorflow as tf
ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='BinaryCrossentropyNoNaN')
class BinaryCrossentropyNoNaN(ks.losses.BinaryCrossentropy):

    def __init__(self, *args, **kwargs):
        super(BinaryCrossentropyNoNaN, self).__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        is_nan = tf.math.is_nan(y_true)
        y_pred = tf.where(is_nan, tf.zeros_like(y_pred), y_pred)
        y_true = tf.where(is_nan, tf.zeros_like(y_true), y_true)
        return super(BinaryCrossentropyNoNaN, self).call(y_true, y_pred)


@ks.utils.register_keras_serializable(package="kgcnn", name="RaggedMeanAbsoluteError")
class RaggedMeanAbsoluteError(ks.losses.Loss):

    def __init__(self, *args, **kwargs):
        super(RaggedMeanAbsoluteError, self).__init__(*args, **kwargs)

    @tf.function
    def call(self, y_true, y_pred):
        return ks.backend.mean(tf.abs(y_pred.flat_values - y_true.flat_values), axis=-1)
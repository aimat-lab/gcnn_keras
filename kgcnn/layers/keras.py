import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.base import KerasLayerWrapperBase

class Dense(KerasLayerWrapperBase):
    """Dense Wrapper Layer to support RaggedTensor input with ragged-rank=1."""

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """Initialize layer as tf.keras.Dense."""
        super(Dense, self).__init__(**kwargs)
        self._kgcnn_wrapper_call_type = 0
        self._kgcnn_wrapper_args = ["units", "activation", "use_bias", "kernel_initializer", "bias_initializer",
                                    "kernel_regularizer", "bias_regularizer", "activity_regularizer",
                                    "kernel_constraint", "bias_constraint"]
        self._kgcnn_wrapper_layer = ks.layers.Dense(units=units, activation=activation,
                                                    use_bias=use_bias, kernel_initializer=kernel_initializer,
                                                    bias_initializer=bias_initializer,
                                                    kernel_regularizer=kernel_regularizer,
                                                    bias_regularizer=bias_regularizer,
                                                    activity_regularizer=activity_regularizer,
                                                    kernel_constraint=kernel_constraint,
                                                    bias_constraint=bias_constraint)


class Activation(KerasLayerWrapperBase):
    """Activation Wrapper Layer to support RaggedTensor input with ragged-rank=1."""

    def __init__(self,
                 activation,
                 activity_regularizer=None,
                 **kwargs):
        """Initialize layer same as tf.keras.Activation."""
        super(Activation, self).__init__(**kwargs)
        self._kgcnn_wrapper_call_type = 0
        self._kgcnn_wrapper_args = ["activation", "activity_regularizer"]
        self._kgcnn_wrapper_layer = tf.keras.layers.Activation(activation=activation,
                                                               activity_regularizer=activity_regularizer)


class Add(KerasLayerWrapperBase):
    """Add Wrapper Layer to support RaggedTensor input with ragged-rank=1."""

    def __init__(self, **kwargs):
        """Initialize layer same as tf.keras.Add."""
        super(Add, self).__init__(**kwargs)
        self._kgcnn_wrapper_call_type = 1
        self._kgcnn_wrapper_args = []
        self._kgcnn_wrapper_layer = ks.layers.Add()


class Average(KerasLayerWrapperBase):
    """Average Wrapper Layer to support RaggedTensor input with ragged-rank=1."""

    def __init__(self, **kwargs):
        """Initialize layer same as tf.keras.Average."""
        super(Average, self).__init__(**kwargs)
        self._kgcnn_wrapper_call_type = 1
        self._kgcnn_wrapper_args = []
        self._kgcnn_wrapper_layer = ks.layers.Average()


class Multiply(KerasLayerWrapperBase):
    """Multiply Wrapper Layer to support RaggedTensor input with ragged-rank=1."""

    def __init__(self, **kwargs):
        """Initialize layer same as tf.keras.Multiply."""
        super(Multiply, self).__init__(**kwargs)
        self._kgcnn_wrapper_call_type = 1
        self._kgcnn_wrapper_args = []
        self._kgcnn_wrapper_layer = ks.layers.Multiply()


class Concatenate(KerasLayerWrapperBase):
    """Concatenate Wrapper Layer to support RaggedTensor input with ragged-rank=1."""

    def __init__(self,
                 axis,
                 **kwargs):
        """Initialize layer same as tf.keras.Concatenate."""
        super(Concatenate, self).__init__(**kwargs)
        self._kgcnn_wrapper_call_type = 1
        self._kgcnn_wrapper_args = ["axis"]
        self._kgcnn_wrapper_layer = ks.layers.Concatenate(axis=axis)


class Dropout(KerasLayerWrapperBase):
    """Dropout Wrapper Layer to support RaggedTensor input with ragged-rank=1."""

    def __init__(self,
                 rate,
                 noise_shape=None,
                 seed=None,
                 **kwargs):
        """Initialize layer same as Activation."""
        super(Dropout, self).__init__(**kwargs)
        self._kgcnn_wrapper_call_type = 0
        self._kgcnn_wrapper_args = ["rate", "noise_shape", "seed"]
        self._kgcnn_wrapper_layer = ks.layers.Dropout(rate=rate, noise_shape=noise_shape, seed=seed)


class LayerNormalization(KerasLayerWrapperBase):
    """LayerNormalization Wrapper Layer to support RaggedTensor input with ragged-rank=1."""

    def __init__(self,
                 axis=-1,
                 epsilon=0.001, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        """Initialize layer same as Activation."""
        super(LayerNormalization, self).__init__(**kwargs)
        self._kgcnn_wrapper_call_type = 0
        self._kgcnn_wrapper_args = ["axis", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer",
                                    "beta_regularizer", "gamma_regularizer", "beta_constraint", "gamma_constraint"]
        self._kgcnn_wrapper_layer = ks.layers.LayerNormalization(axis=axis, epsilon=epsilon, center=center, scale=scale,
                                                                 beta_initializer=beta_initializer,
                                                                 gamma_initializer=gamma_initializer,
                                                                 beta_regularizer=beta_regularizer,
                                                                 gamma_regularizer=gamma_regularizer,
                                                                 beta_constraint=beta_constraint,
                                                                 gamma_constraint=gamma_constraint)

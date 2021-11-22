import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.base import KerasWrapperBase, GraphBaseLayer

# There are limitations for RaggedTensor working with standard Keras layers. Here are some simple wrappers.
# This is a temporary solution until future versions of TensorFlow support more RaggedTensor arguments.
# Since all kgcnn layers work with ragged_rank=1 and defined inner dimension. This case can be caught explicitly.
from kgcnn.ops.axis import get_positive_axis


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='Dense')
class Dense(KerasWrapperBase):

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


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='Activation')
class Activation(KerasWrapperBase):

    def __init__(self,
                 activation,
                 activity_regularizer=None,
                 **kwargs):
        """Initialize layer same as tf.keras.Activation."""
        super(Activation, self).__init__(**kwargs)
        self._kgcnn_wrapper_args = ["activation", "activity_regularizer"]
        self._kgcnn_wrapper_layer = tf.keras.layers.Activation(activation=activation,
                                                               activity_regularizer=activity_regularizer)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='Add')
class Add(KerasWrapperBase):

    def __init__(self, **kwargs):
        """Initialize layer same as tf.keras.Add."""
        super(Add, self).__init__(**kwargs)
        self._kgcnn_wrapper_args = []
        self._kgcnn_wrapper_layer = ks.layers.Add()


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='Subtract')
class Subtract(KerasWrapperBase):

    def __init__(self, **kwargs):
        """Initialize layer same as tf.keras.Add."""
        super(Subtract, self).__init__(**kwargs)
        self._kgcnn_wrapper_args = []
        self._kgcnn_wrapper_layer = ks.layers.Subtract()


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='Average')
class Average(KerasWrapperBase):

    def __init__(self, **kwargs):
        """Initialize layer same as tf.keras.Average."""
        super(Average, self).__init__(**kwargs)
        self._kgcnn_wrapper_args = []
        self._kgcnn_wrapper_layer = ks.layers.Average()


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='Multiply')
class Multiply(KerasWrapperBase):

    def __init__(self, **kwargs):
        """Initialize layer same as tf.keras.Multiply."""
        super(Multiply, self).__init__(**kwargs)
        self._kgcnn_wrapper_args = []
        self._kgcnn_wrapper_layer = ks.layers.Multiply()


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='Dropout')
class Dropout(KerasWrapperBase):

    def __init__(self,
                 rate,
                 noise_shape=None,
                 seed=None,
                 **kwargs):
        """Initialize layer same as Activation."""
        super(Dropout, self).__init__(**kwargs)
        self._kgcnn_wrapper_args = ["rate", "noise_shape", "seed"]
        self._kgcnn_wrapper_layer = ks.layers.Dropout(rate=rate, noise_shape=noise_shape, seed=seed)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='Concatenate')
class Concatenate(KerasWrapperBase):

    def __init__(self,
                 axis=-1,
                 **kwargs):
        super(Concatenate, self).__init__(**kwargs)
        self._kgcnn_wrapper_args = ["axis"]
        self._kgcnn_wrapper_layer = ks.layers.Concatenate(axis=axis)

    def call(self, inputs, **kwargs):
        """Forward pass wrapping tf.keras layer."""
        # Simply wrapper for self._kgcnn_wrapper_layer. Only works for simply element-wise operations.
        if all([isinstance(x, tf.RaggedTensor) for x in inputs]):
            # However, partition could be different, so this is only okay if ragged_validate=False
            # For defined inner-dimension and raggd_rank=1 can do sloppy concatenate on values.
            if all([x.ragged_rank == 1 for x in inputs]) and self._kgcnn_wrapper_layer.axis == -1 and all(
                    [x.shape[-1] is not None for x in inputs]):
                out = self._kgcnn_wrapper_layer([x.values for x in inputs], **kwargs)  # will be all Tensor
                out = tf.RaggedTensor.from_row_splits(out, inputs[0].row_splits, validate=self.ragged_validate)
                return out
            else:
                print("WARNING: Layer", self.name, "fail call on values for ragged_rank=1, attempting keras call... ")
        # Try normal keras call
        return self._kgcnn_wrapper_layer(inputs, **kwargs)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LayerNormalization')
class LayerNormalization(KerasWrapperBase):

    def __init__(self,
                 axis=-1,
                 epsilon=1e-3, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        """Initialize layer same as Activation."""
        super(LayerNormalization, self).__init__(**kwargs)
        self.axis = axis  # We do not change the axis here
        self._kgcnn_wrapper_args = ["axis", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer",
                                    "beta_regularizer", "gamma_regularizer", "beta_constraint", "gamma_constraint"]
        self._kgcnn_wrapper_layer = ks.layers.LayerNormalization(axis=axis, epsilon=epsilon,
                                                                 center=center, scale=scale,
                                                                 beta_initializer=beta_initializer,
                                                                 gamma_initializer=gamma_initializer,
                                                                 beta_regularizer=beta_regularizer,
                                                                 gamma_regularizer=gamma_regularizer,
                                                                 beta_constraint=beta_constraint,
                                                                 gamma_constraint=gamma_constraint, dtype="float32")
        if self.axis != -1:
            print("WARNING: This implementation only supports axis=-1 for RaggedTensors for now.")

    def call(self, inputs, **kwargs):
        """Forward pass wrapping tf.keras layer."""
        if isinstance(inputs, tf.RaggedTensor):
            if self.axis == -1 and inputs.shape[-1] is not None and inputs.ragged_rank == 1:
                value_tensor = inputs.values  # will be Tensor
                out_tensor = self._kgcnn_wrapper_layer(value_tensor, **kwargs)
                return tf.RaggedTensor.from_row_splits(out_tensor, inputs.row_splits, validate=self.ragged_validate)
            else:
                print("WARNING: Layer", self.name, "fail call on values for ragged_rank=1, attempting keras call... ")
        # Try normal keras call
        return self._kgcnn_wrapper_layer(inputs, **kwargs)

    def get_config(self):
        config = super(LayerNormalization, self).get_config()
        config.update({"axis": self.axis})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='BatchNormalization')
class BatchNormalization(KerasWrapperBase):

    def __init__(self,
                 axis=-1,
                 momentum=0.99, epsilon=0.001, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones', beta_regularizer=None,
                 gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                 **kwargs):
        """Initialize layer same as Activation."""
        super(BatchNormalization, self).__init__(**kwargs)
        self.axis = axis  # We do not change the axis here (just as input)
        self._kgcnn_wrapper_args = ["axis", "momentum", "epsilon", "scale", "beta_initializer", "gamma_initializer",
                                    "beta_regularizer", "gamma_regularizer", "beta_constraint", "gamma_constraint"]
        self._kgcnn_wrapper_layer = ks.layers.BatchNormalization(axis=axis, momentum=momentum, epsilon=epsilon,
                                                                 center=center, scale=scale,
                                                                 beta_initializer=beta_initializer,
                                                                 gamma_initializer=gamma_initializer,
                                                                 moving_mean_initializer=moving_mean_initializer,
                                                                 moving_variance_initializer=moving_variance_initializer,
                                                                 beta_regularizer=beta_regularizer,
                                                                 gamma_regularizer=gamma_regularizer,
                                                                 beta_constraint=beta_constraint,
                                                                 gamma_constraint=gamma_constraint)
        if self.axis != -1:
            print("WARNING: This implementation only supports axis=-1 for RaggedTensors for now.")

    def call(self, inputs, **kwargs):
        """Forward pass wrapping tf.keras layer."""
        if isinstance(inputs, tf.RaggedTensor):
            if self.axis == -1 and inputs.shape[-1] is not None and inputs.ragged_rank == 1:
                value_tensor = inputs.values  # will be Tensor
                out_tensor = self._kgcnn_wrapper_layer(value_tensor, **kwargs)
                return tf.RaggedTensor.from_row_splits(out_tensor, inputs.row_splits, validate=self.ragged_validate)
            else:
                print("WARNING: Layer", self.name, "fail call on values for ragged_rank=1, attempting keras call... ")
        # Try normal keras call
        return self._kgcnn_wrapper_layer(inputs, **kwargs)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        config.update({"axis": self.axis})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ExpandDims')
class ExpandDims(GraphBaseLayer):

    def __init__(self,
                 axis=-1,
                 **kwargs):
        """Initialize layer same as Activation."""
        super(ExpandDims, self).__init__(**kwargs)
        self.axis = axis  # We do not change the axis here

    def call(self, inputs, **kwargs):
        """Forward pass wrapping tf.keras layer."""
        if isinstance(inputs, tf.RaggedTensor):
            axis = get_positive_axis(self.axis, inputs.shape.rank + 1)
            if axis > 1 and inputs.ragged_rank == 1:
                value_tensor = inputs.values  # will be Tensor
                out_tensor = tf.expand_dims(value_tensor, axis=axis - 1)
                return tf.RaggedTensor.from_row_splits(out_tensor, inputs.row_splits, validate=self.ragged_validate)
            else:
                print("WARNING: Layer", self.name, "fail call on values for ragged_rank=1, attempting keras call... ")
        # Try normal operation
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super(GraphBaseLayer, self).get_config()
        config.update({"axis": self.axis})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ZerosLike')
class ZerosLike(GraphBaseLayer):
    """Make a zero-like graph tensor. Calls tf.zeros_like()."""

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(ZerosLike, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(ZerosLike, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.RaggedTensor): Tensor of node or edge embeddings of shape (batch, [N], F)

        Returns:
            tf.RaggedTensor: Zero-like tensor of input.
        """
        if isinstance(inputs, tf.RaggedTensor):
            if inputs.ragged_rank == 1:
                zero_tensor = tf.zeros_like(inputs.values)  # will be Tensor
                return tf.RaggedTensor.from_row_splits(zero_tensor, inputs.row_splits, validate=self.ragged_validate)
            else:
                print("WARNING: Layer", self.name, "fail call on values for ragged_rank=1.")
        # Try normal tf call
        return tf.zeros_like(inputs)
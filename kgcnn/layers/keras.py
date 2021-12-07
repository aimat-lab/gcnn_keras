import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.base import GraphBaseLayer

# There are limitations for RaggedTensor working with standard Keras layers. Here are some simple wrappers.
# This is a temporary solution until future versions of TensorFlow support more RaggedTensor arguments.
# Since all kgcnn layers work with ragged_rank=1 and defined inner dimension. This case can be caught explicitly.
from kgcnn.ops.axis import get_positive_axis


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='Dense')
class DenseEmbedding(GraphBaseLayer):

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
        super(DenseEmbedding, self).__init__(**kwargs)
        self._layer_dense = ks.layers.Dense(units=units, activation=activation,
                                            use_bias=use_bias, kernel_initializer=kernel_initializer,
                                            bias_initializer=bias_initializer,
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer,
                                            activity_regularizer=activity_regularizer,
                                            kernel_constraint=kernel_constraint,
                                            bias_constraint=bias_constraint)
        self._add_layer_config_to_self = {
            "_layer_dense": ["units", "activation", "use_bias", "kernel_initializer", "bias_initializer",
                             "kernel_regularizer", "bias_regularizer", "activity_regularizer",
                             "kernel_constraint", "bias_constraint"]}

    def call(self, inputs, **kwargs):
        """Forward pass wrapping tf.keras.layers"""
        # For Dense can call on flat values too.
        if isinstance(inputs, tf.RaggedTensor):
            return tf.ragged.map_flat_values(self._layer_dense, inputs, **kwargs)
        return self._layer_dense(inputs, **kwargs)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ActivationEmbedding')
class ActivationEmbedding(GraphBaseLayer):

    def __init__(self,
                 activation,
                 activity_regularizer=None,
                 **kwargs):
        """Initialize layer."""
        super(ActivationEmbedding, self).__init__(**kwargs)
        self._layer_act = tf.keras.layers.Activation(activation=activation, activity_regularizer=activity_regularizer)
        self._add_layer_config_to_self = {"_layer_act": ["activation", "activity_regularizer"]}

    def call(self, inputs, **kwargs):
        """Forward pass corresponding to keras Activation layer."""
        return self.call_on_ragged_values(self._layer_act, inputs, **kwargs)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LazyAdd')
class LazyAdd(GraphBaseLayer):

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(LazyAdd, self).__init__(**kwargs)
        self._layer_add = ks.layers.Add()

    def call(self, inputs, **kwargs):
        """Forward pass corresponding to keras Add layer."""
        return self.call_on_ragged_values(self._layer_add, inputs, **kwargs)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LazySubtract')
class LazySubtract(GraphBaseLayer):

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(LazySubtract, self).__init__(**kwargs)
        self._layer_subtract = ks.layers.Subtract()

    def call(self, inputs, **kwargs):
        """Forward pass corresponding to keras Average layer."""
        return self.call_on_ragged_values(self._layer_subtract, inputs, **kwargs)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LazyAverage')
class LazyAverage(GraphBaseLayer):

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(LazyAverage, self).__init__(**kwargs)
        self._layer_avg = ks.layers.Average()

    def call(self, inputs, **kwargs):
        """Forward pass corresponding to keras Average layer."""
        return self.call_on_ragged_values(self._layer_avg, inputs, **kwargs)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LazyMultiply')
class LazyMultiply(GraphBaseLayer):

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(LazyMultiply, self).__init__(**kwargs)
        self._layer_mult = ks.layers.Multiply()

    def call(self, inputs, **kwargs):
        """Forward pass corresponding to keras Multiply layer."""
        return self.call_on_ragged_values(self._layer_mult, inputs, **kwargs)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='Dropout')
class Dropout(GraphBaseLayer):

    def __init__(self,
                 rate,
                 noise_shape=None,
                 seed=None,
                 **kwargs):
        """Initialize layer same as Activation."""
        super(Dropout, self).__init__(**kwargs)
        self._layer_drop = ks.layers.Dropout(rate=rate, noise_shape=noise_shape, seed=seed)
        self._add_layer_config_to_self = {"_layer_drop": ["rate", "noise_shape", "seed"]}

    def call(self, inputs, **kwargs):
        """Forward pass corresponding to keras Dropout layer."""
        return self.call_on_ragged_values(self._layer_drop, inputs, **kwargs)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LazyConcatenate')
class LazyConcatenate(GraphBaseLayer):

    def __init__(self,
                 axis=-1,
                 **kwargs):
        super(LazyConcatenate, self).__init__(**kwargs)
        self._layer_concat = ks.layers.Concatenate(axis=axis)
        self._add_layer_config_to_self = {"_layer_concat": ["axis"]}

    def call(self, inputs, **kwargs):
        """Forward pass wrapping tf.keras layer."""
        # Simply wrapper for self._kgcnn_wrapper_layer. Only works for simply element-wise operations.
        if all([isinstance(x, tf.RaggedTensor) for x in inputs]):
            # However, partition could be different, so this is only okay if ragged_validate=False
            # For defined inner-dimension and raggd_rank=1 can do sloppy concatenate on values.
            if all([x.ragged_rank == 1 for x in inputs]) and self._layer_concat.axis == -1 and all(
                    [x.shape[-1] is not None for x in inputs]):
                out = self._layer_concat([x.values for x in inputs], **kwargs)  # will be all Tensor
                out = tf.RaggedTensor.from_row_splits(out, inputs[0].row_splits, validate=self.ragged_validate)
                return out
            else:
                print("WARNING: Layer", self.name, "fail call on values for ragged_rank=1, attempting keras call... ")
        # Try normal keras call
        return self._layer_concat(inputs, **kwargs)


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
        return self.call_on_ragged_values(tf.zeros_like, inputs)

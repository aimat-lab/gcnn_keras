import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.axis import get_positive_axis


# There are limitations for RaggedTensor working with standard Keras layers. Here are some simple surrogates.
# This is a temporary solution until future versions of TensorFlow support more RaggedTensor arguments.
# Since most kgcnn layers work with ragged_rank = 1 and defined inner dimension, this case can be caught explicitly.


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='DenseEmbedding')
class DenseEmbedding(GraphBaseLayer):
    r"""Dense layer for ragged tensors representing a geometric or graph tensor such as node or edge embeddings.
    Current tensorflow version now support ragged input for :obj:`tf.keras.layers.Dense` with defined inner dimension.
    This layer is kept for backward compatibility and but does not necessarily have to be used in models anymore.
    """

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
        """Forward pass using Dense on flat values."""
        # For Dense can call on flat values.
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
        return self.call_on_values_tensor_of_ragged(self._layer_act, inputs, **kwargs)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LazyAdd')
class LazyAdd(GraphBaseLayer):

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(LazyAdd, self).__init__(**kwargs)
        self._layer_add = ks.layers.Add()

    def call(self, inputs, **kwargs):
        """Forward pass corresponding to keras Add layer."""
        return self.call_on_values_tensor_of_ragged(self._layer_add, inputs, **kwargs)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LazySubtract')
class LazySubtract(GraphBaseLayer):

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(LazySubtract, self).__init__(**kwargs)
        self._layer_subtract = ks.layers.Subtract()

    def call(self, inputs, **kwargs):
        """Forward pass corresponding layer."""
        return self.call_on_values_tensor_of_ragged(self._layer_subtract, inputs, **kwargs)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LazyAverage')
class LazyAverage(GraphBaseLayer):

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(LazyAverage, self).__init__(**kwargs)
        self._layer_avg = ks.layers.Average()

    def call(self, inputs, **kwargs):
        """Forward pass corresponding to keras Average layer."""
        return self.call_on_values_tensor_of_ragged(self._layer_avg, inputs, **kwargs)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LazyMultiply')
class LazyMultiply(GraphBaseLayer):

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(LazyMultiply, self).__init__(**kwargs)
        self._layer_mult = ks.layers.Multiply()

    def call(self, inputs, **kwargs):
        """Forward pass corresponding to keras Multiply layer."""
        return self.call_on_values_tensor_of_ragged(self._layer_mult, inputs, **kwargs)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='DropoutEmbedding')
class DropoutEmbedding(GraphBaseLayer):

    def __init__(self,
                 rate,
                 noise_shape=None,
                 seed=None,
                 **kwargs):
        """Initialize layer same as Activation."""
        super(DropoutEmbedding, self).__init__(**kwargs)
        self._layer_drop = ks.layers.Dropout(rate=rate, noise_shape=noise_shape, seed=seed)
        self._add_layer_config_to_self = {"_layer_drop": ["rate", "noise_shape", "seed"]}

    def call(self, inputs, **kwargs):
        """Forward pass corresponding to keras Dropout layer."""
        return self.call_on_values_tensor_of_ragged(self._layer_drop, inputs, **kwargs)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='LazyConcatenate')
class LazyConcatenate(GraphBaseLayer):

    def __init__(self,
                 axis=-1,
                 **kwargs):
        """Initialize layer."""
        super(LazyConcatenate, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        """Build layer from input shape."""
        super(LazyConcatenate, self).build(input_shape)
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 1:
            raise ValueError(
                'A `Concatenate` layer should be called on a list of '
                f'at least 1 input. Received: input_shape={input_shape}')
        if all(shape is None for shape in input_shape):
            return
        # Make sure all the shapes have same ranks.
        ranks = set(len(shape) for shape in input_shape)
        if len(ranks) == 1:
            # Make axis positive then for call on values.
            self.axis = get_positive_axis(self.axis, len(input_shape[0]))

    def call(self, inputs, **kwargs):
        """Forward pass. Concatenate possibly ragged tensors.

        Args:
            inputs (list): List of tensors to concatenate.

        Returns:
            tf.tensor: Single concatenated tensor.
        """
        return self.call_on_values_tensor_of_ragged(tf.concat, inputs, axis=self.axis)

    def get_config(self):
        config = super(LazyConcatenate, self).get_config()
        config.update({"axis": self.axis})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ExpandDims')
class ExpandDims(GraphBaseLayer):

    def __init__(self,
                 axis=-1,
                 **kwargs):
        """Initialize layer."""
        super(ExpandDims, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        super(ExpandDims, self).build(input_shape)
        # If rank is not defined can't call on values, if axis does not happen to be positive.
        if len(input_shape) == 0:
            return
        # The possible target axis can be one rank larger to increase rank with expand_dims.
        self.axis = get_positive_axis(self.axis, len(input_shape) + 1)

    def call(self, inputs, **kwargs):
        """Forward pass wrapping tf.keras layer."""
        return self.call_on_values_tensor_of_ragged(tf.expand_dims, inputs, axis=self.axis)

    def get_config(self):
        config = super(ExpandDims, self).get_config()
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
        return self.call_on_values_tensor_of_ragged(tf.zeros_like, inputs)


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='ReduceSum')
class ReduceSum(GraphBaseLayer):
    """Make a zero-like graph tensor. Calls tf.zeros_like()."""

    def __init__(self, axis=-1, **kwargs):
        """Initialize layer."""
        super(ReduceSum, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        """Build layer."""
        super(ReduceSum, self).build(input_shape)
        # If rank is not defined can't call on values, if axis does not happen to be positive.
        if len(input_shape) == 0:
            return
        # Set axis to be positive for defined rank to call on values.
        self.axis = get_positive_axis(self.axis, len(input_shape))

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.RaggedTensor): Tensor of node or edge embeddings of shape (batch, [N], F)

        Returns:
            tf.RaggedTensor: Zero-like tensor of input.
        """
        return self.call_on_values_tensor_of_ragged(tf.reduce_sum, inputs, axis=self.axis)

    def get_config(self):
        config = super(ReduceSum, self).get_config()
        config.update({"axis": self.axis})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='OptionalInputEmbedding')
class OptionalInputEmbedding(GraphBaseLayer):
    """Optional Embedding layer."""

    def __init__(self,
                 input_dim,
                 output_dim,
                 use_embedding=False,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        """Initialize layer."""
        super(OptionalInputEmbedding, self).__init__(**kwargs)
        self.use_embedding = use_embedding

        if use_embedding:
            self._layer_embed = ks.layers.Embedding(input_dim=input_dim, output_dim=output_dim,
                                                    embeddings_initializer=embeddings_initializer,
                                                    embeddings_regularizer=embeddings_regularizer,
                                                    activity_regularizer=activity_regularizer,
                                                    embeddings_constraint=embeddings_constraint,
                                                    mask_zero=mask_zero, input_length=input_length)
            self._add_layer_config_to_self = {"_layer_embed": ["input_dim", "output_dim", "embeddings_initializer",
                                                               "embeddings_regularizer", "activity_regularizer",
                                                               "embeddings_constraint", "mask_zero", "input_length"]}

    def build(self, input_shape):
        """Build layer."""
        super(OptionalInputEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.RaggedTensor): Tensor of number or embeddings of shape (batch, [N]) or (batch, [N], F)

        Returns:
            tf.RaggedTensor: Zero-like tensor of input.
        """
        if self.use_embedding:
            return self._layer_embed(inputs)
        return inputs

    def get_config(self):
        config = super(OptionalInputEmbedding, self).get_config()
        config.update({"use_embedding": self.use_embedding})
        return config

import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.axis import get_positive_axis
ks = tf.keras  # import tensorflow.keras as ks

# There are limitations for RaggedTensor working with standard Keras layers, but which are successively reduced with
# more recent tensorflow versions (tf-version >= 2.2).
# For backward compatibility we keep keras layer replacements to work with RaggedTensor in this module.
# For example with tf-version==2.8, DenseEmbedding is equivalent to ks.layers.Dense.
# Note that here are LazyAdd and LazyConcatenate etc. layers which are slightly different from keras layer, which also
# work on RaggedTensor, but neglect shape check if 'ragged_validate' is set to False.


@ks.utils.register_keras_serializable(package='kgcnn', name='DenseEmbedding')
class DenseEmbedding(GraphBaseLayer):
    r"""Dense layer for ragged tensors representing a geometric or graph tensor such as node or edge embeddings.
    Latest tensorflow version now support ragged input for :obj:`ks.layers.Dense` with defined inner dimension.
    This layer is kept for backward compatibility but does not necessarily have to be used in models anymore.
    A :obj:`DenseEmbedding` layer computes a densely-connected NN layer, i.e. a linear transformation of the input
    :math:`\mathbf{x}` with the kernel weights matrix :math:`\mathbf{W}` and bias :math:`\mathbf{b}`
    plus (possibly non-linear) activation function :math:`\sigma`.

    .. math::
        \mathbf{x}' = \sigma (\mathbf{x} \mathbf{W} + \mathbf{b})

    """

    def __init__(self,
                 units: int,
                 activation=None,
                 use_bias: bool = True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        r"""Initialize layer like :obj:`ks.layers.Dense`.

         Args:
            units: Positive integer, dimensionality of the output space.
            activation: Activation function to use.
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix.
            bias_initializer: Initializer for the bias vector.
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation").
            kernel_constraint: Constraint function applied to
                the `kernel` weights matrix.
            bias_constraint: Constraint function applied to the bias vector.
        """
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
        r"""Forward pass using :obj:`ks.layers.Dense` on :obj:`map_flat_values`.

        Args:
            inputs (tf.RaggedTensor): Input tensor with last dimension must be defined, e.g. not be `None`.

        Returns:
            tf.RaggedTensor: NN output N-D tensor with shape: `(batch_size, ..., units)`.
        """
        # For Dense can call on flat values.
        if isinstance(inputs, tf.RaggedTensor):
            return tf.ragged.map_flat_values(self._layer_dense, inputs, **kwargs)
        # Else try call dense layer directly.
        return self._layer_dense(inputs, **kwargs)


@ks.utils.register_keras_serializable(package='kgcnn', name='ActivationEmbedding')
class ActivationEmbedding(GraphBaseLayer):
    r"""Activation layer for ragged tensors representing a geometric or graph tensor such as node or edge embeddings.
    A :obj:`ActivationEmbedding` applies an activation function to an output, i.e. a transformation of the input
    :math:`\mathbf{x}` via activation function :math:`\sigma`.

    .. math::
        \mathbf{x}' = \sigma (\mathbf{x})

    """

    def __init__(self,
                 activation,
                 activity_regularizer=None,
                 **kwargs):
        """Initialize layer.

        Args:
            activation: Activation function, such as `tf.nn.relu`, or string name of
                built-in activation function, such as "relu".
            activity_regularizer: Regularizer function applied to the output of the layer (its "activation").
        """
        super(ActivationEmbedding, self).__init__(**kwargs)
        self._layer_act = ks.layers.Activation(activation=activation, activity_regularizer=activity_regularizer)
        self._add_layer_config_to_self = {"_layer_act": ["activation", "activity_regularizer"]}

    def call(self, inputs, **kwargs):
        r"""Forward pass corresponding to keras :obj:`Activation` layer.

        Args:
            inputs (tf.RaggedTensor): Input tensor of arbitrary shape.

        Returns:
            tf.RaggedTensor: Output tensor with activation applied.
        """
        return self.call_on_values_tensor_of_ragged(self._layer_act, inputs, **kwargs)


@ks.utils.register_keras_serializable(package='kgcnn', name='LazyAdd')
class LazyAdd(GraphBaseLayer):
    r"""Layer that adds a list of inputs of e.g. geometric or graph tensor such as node or edge embeddings.
    It takes as input a list of tensors, all the same shape, and returns a single tensor (also of the same shape).
    For :obj:`RaggedTensor` the addition is directly performed on the `values` tensor of the ragged
    input if all tensor in the list have `ragged_rank=1` and if `ragged_validate` is set to `False`.
    Apart from debugging, this can imply a significant performance boost if ragged shape checks can be avoided.

    .. math::
        \mathbf{x}' = \sum_i \; \mathbf{x}_i
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(LazyAdd, self).__init__(**kwargs)
        self._layer_add = ks.layers.Add()

    def call(self, inputs, **kwargs):
        r"""Forward pass corresponding to keras :obj:`Add` layer.

        Args:
            inputs (list): List of input tensor of same shape.

        Returns:
            tf.RaggedTensor: Single output tensor with same shape.
        """
        return self.call_on_values_tensor_of_ragged(self._layer_add, inputs, **kwargs)


@ks.utils.register_keras_serializable(package='kgcnn', name='LazySubtract')
class LazySubtract(GraphBaseLayer):
    r"""Layer that subtracts two inputs of e.g. geometric or graph tensor such as node or edge embeddings.
    It takes as input a list of two tensors, both the same shape, and returns a single tensor (also of the same shape).
    For :obj:`RaggedTensor` the subtraction is directly performed on the `values` tensor of the ragged
    input if both tensor in the list have `ragged_rank=1` and if `ragged_validate` is set to `False`.
    Apart from debugging, this can imply a significant performance boost if ragged shape checks can be avoided.

    .. math::
        \mathbf{x}' = \mathbf{x}_0 - \mathbf{x}_1
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(LazySubtract, self).__init__(**kwargs)
        self._layer_subtract = ks.layers.Subtract()

    def call(self, inputs, **kwargs):
        r"""Forward pass corresponding to keras :obj:`Subtract` layer.

        Args:
            inputs (list): List of two input tensor of same shape.

        Returns:
            tf.RaggedTensor: Single output tensor which is (inputs[0] - inputs[1]).
        """
        return self.call_on_values_tensor_of_ragged(self._layer_subtract, inputs, **kwargs)


@ks.utils.register_keras_serializable(package='kgcnn', name='LazyAverage')
class LazyAverage(GraphBaseLayer):
    r"""Layer that averages a list of inputs element-wise of e.g. geometric or graph tensor such as node or
    edge embeddings.
    It takes as input a list of tensors, all the same shape, and returns a single tensor (also of the same shape).
    For :obj:`RaggedTensor` the average is directly performed on the `values` tensor of the ragged
    input if all tensor in the list have `ragged_rank=1` and if `ragged_validate` is set to `False`.
    Apart from debugging, this can imply a significant performance boost if ragged shape checks can be avoided.

    .. math::
        \mathbf{x}' = \frac{1}{N} \sum_{i=0,\dots,N} \;\; \mathbf{x}_i
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(LazyAverage, self).__init__(**kwargs)
        self._layer_avg = ks.layers.Average()

    def call(self, inputs, **kwargs):
        r"""Forward pass corresponding to keras :obj:`Average` layer.

        Args:
            inputs (list): List of input tensor of same shape.

        Returns:
            tf.RaggedTensor: Single output tensor with same shape.
        """
        return self.call_on_values_tensor_of_ragged(self._layer_avg, inputs, **kwargs)


@ks.utils.register_keras_serializable(package='kgcnn', name='LazyMultiply')
class LazyMultiply(GraphBaseLayer):

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(LazyMultiply, self).__init__(**kwargs)
        self._layer_mult = ks.layers.Multiply()

    def call(self, inputs, **kwargs):
        """Forward pass corresponding to keras Multiply layer."""
        return self.call_on_values_tensor_of_ragged(self._layer_mult, inputs, **kwargs)


@ks.utils.register_keras_serializable(package='kgcnn', name='DropoutEmbedding')
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


@ks.utils.register_keras_serializable(package='kgcnn', name='LazyConcatenate')
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


@ks.utils.register_keras_serializable(package='kgcnn', name='ExpandDims')
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
        """Forward pass wrapping ks layer."""
        return self.call_on_values_tensor_of_ragged(tf.expand_dims, inputs, axis=self.axis)

    def get_config(self):
        config = super(ExpandDims, self).get_config()
        config.update({"axis": self.axis})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='ZerosLike')
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


@ks.utils.register_keras_serializable(package='kgcnn', name='ReduceSum')
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


@ks.utils.register_keras_serializable(package='kgcnn', name='OptionalInputEmbedding')
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

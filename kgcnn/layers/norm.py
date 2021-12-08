import tensorflow as tf
from tensorflow import keras as ks

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.axis import get_positive_axis


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GraphLayerNormalization')
class GraphLayerNormalization(GraphBaseLayer):

    def __init__(self,
                 axis=-1,
                 epsilon=1e-3, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        """Initialize layer same as Activation."""
        super(GraphLayerNormalization, self).__init__(**kwargs)
        # The axis 0,1 are merged for ragged embedding input.
        if isinstance(axis, (list, tuple)):
            self.axis = list(axis[:])
            if any([x == 0 for x in self.axis]):
                raise ValueError("Positive axis for graph normalization must be > 0 or negative.")
            axis_values = [x - 1 if x > 0 else x for x in self.axis]
        elif isinstance(axis, int):
            self.axis = axis
            if self.axis == 0:
                raise ValueError("Positive axis for graph normalization must be > 0 or negative.")
            axis_values = axis - 1 if axis > 0 else axis
        else:
            raise TypeError('Expected an int or a list/tuple of ints for the '
                            'argument \'axis\', but received: %r' % axis)

        self._layer_norm = ks.layers.LayerNormalization(axis=axis_values, epsilon=epsilon,
                                                        center=center, scale=scale,
                                                        beta_initializer=beta_initializer,
                                                        gamma_initializer=gamma_initializer,
                                                        beta_regularizer=beta_regularizer,
                                                        gamma_regularizer=gamma_regularizer,
                                                        beta_constraint=beta_constraint,
                                                        gamma_constraint=gamma_constraint, dtype=self.dtype)
        # We can add config from keras layer except the axis parameter.
        self._add_layer_config_to_self = {"_layer_norm": ["epsilon", "center", "scale", "beta_initializer",
                                                          "gamma_initializer", "beta_regularizer", "gamma_regularizer",
                                                          "beta_constraint", "gamma_constraint"]}

    def build(self, input_shape):
        """Build layer."""
        super(GraphLayerNormalization, self).build(input_shape)
        n_dims = len(input_shape)
        if isinstance(self.axis, int):
            axis = get_positive_axis(self.axis, n_dims)
            if axis < 1:
                raise ValueError("The (positive) axis must be > 0.")
            axis_values = axis - 1
        elif isinstance(self.axis, list):
            axis = [get_positive_axis(x, n_dims) for x in self.axis]
            if any([x < 1 for x in axis]):
                raise ValueError("All (positive) axis must be > 0.")
            axis_values = [x - 1 for x in axis]
        else:
            raise TypeError("Expected an int or a list of ints for the axis %s" % self.axis)
        # Give positive axis to self, after built the axis is always positive
        self.axis = axis
        # Remove batch dimension as we will call directly on value tensor in call.
        self._layer_norm.axis = axis_values

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            tf.RaggedTensor: Embeddings of shape (batch, [M], F, ...)

        Returns:
            tf.RaggedTensor: Normalized ragged tensor of identical shape (batch, [M], F, ...)
        """
        self.assert_ragged_input_rank(inputs, ragged_rank=1)  # Must have ragged input here for correct axis.
        return self.call_on_values_tensor_of_ragged(self._layer_norm, inputs, **kwargs)

    def get_config(self):
        config = super(GraphLayerNormalization, self).get_config()
        config.update({"axis": self.axis})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GraphBatchNormalization')
class GraphBatchNormalization(GraphBaseLayer):

    def __init__(self,
                 axis=-1,
                 momentum=0.99, epsilon=0.001, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones', beta_regularizer=None,
                 gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                 **kwargs):
        """Initialize layer same as Activation."""
        super(GraphBatchNormalization, self).__init__(**kwargs)
        # The axis 0,1 are merged for ragged embedding input.
        if isinstance(axis, (list, tuple)):
            self.axis = list(axis[:])
            if any([x == 0 for x in self.axis]):
                raise ValueError("Positive axis for graph normalization must be > 0 or negative.")
            axis_values = [x - 1 if x > 0 else x for x in self.axis]
        elif isinstance(axis, int):
            self.axis = axis
            if self.axis == 0:
                raise ValueError("Positive axis for graph normalization must be > 0 or negative.")
            axis_values = axis - 1 if axis > 0 else axis
        else:
            raise TypeError('Expected an int or a list/tuple of ints for the '
                            'argument \'axis\', but received: %r' % axis)

        self._layer_norm = ks.layers.BatchNormalization(axis=axis_values, momentum=momentum, epsilon=epsilon,
                                                        center=center, scale=scale,
                                                        beta_initializer=beta_initializer,
                                                        gamma_initializer=gamma_initializer,
                                                        moving_mean_initializer=moving_mean_initializer,
                                                        moving_variance_initializer=moving_variance_initializer,
                                                        beta_regularizer=beta_regularizer,
                                                        gamma_regularizer=gamma_regularizer,
                                                        beta_constraint=beta_constraint,
                                                        gamma_constraint=gamma_constraint)
        # We can add config from keras layer except the axis parameter.
        self._add_layer_config_to_self = {
            "_layer_norm": ["momentum", "epsilon", "scale", "center", "beta_initializer", "gamma_initializer",
                            "moving_mean_initializer", "moving_variance_initializer"
                            "beta_regularizer", "gamma_regularizer", "beta_constraint", "gamma_constraint"]}

    def build(self, input_shape):
        """Build layer."""
        super(GraphBatchNormalization, self).build(input_shape)
        n_dims = len(input_shape)
        if isinstance(self.axis, int):
            axis = get_positive_axis(self.axis, n_dims)
            if axis < 1:
                raise ValueError("The (positive) axis must be > 0.")
            axis_values = axis - 1
        elif isinstance(self.axis, list):
            axis = [get_positive_axis(x, n_dims) for x in self.axis]
            if any([x < 1 for x in axis]):
                raise ValueError("All (positive) axis must be > 0.")
            axis_values = [x - 1 for x in axis]
        else:
            raise TypeError("Expected an int or a list of ints for the axis %s" % self.axis)
        # Give positive axis to self, after built the axis is always positive
        self.axis = axis
        # Remove batch dimension as we will call directly on value tensor in call.
        self._layer_norm.axis = axis_values

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            tf.RaggedTensor: Embeddings of shape (batch, [M], F, ...)

        Returns:
            tf.RaggedTensor: Normalized ragged tensor of identical shape (batch, [M], F, ...)
        """
        self.assert_ragged_input_rank(inputs, ragged_rank=1)  # Must have ragged input here for correct axis.
        return self.call_on_values_tensor_of_ragged(self._layer_norm, inputs, **kwargs)

    def get_config(self):
        config = super(GraphBatchNormalization, self).get_config()
        config.update({"axis": self.axis})
        return config

# test = tf.RaggedTensor.from_row_lengths(tf.constant([[[0.0],[0.0]], [[1.0],[1.0]], [[2.0],[2.0]]]),tf.constant([2,1]))

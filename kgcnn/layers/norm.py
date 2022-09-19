import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.axis import get_positive_axis
ks = tf.keras


@ks.utils.register_keras_serializable(package='kgcnn', name='GraphLayerNormalization')
class GraphLayerNormalization(GraphBaseLayer):
    r"""Graph Layer normalization for (ragged) graph tensor objects.

    Uses `ks.layers.LayerNormalization` on all node or edge features in a batch.
    Following convention suggested by `GraphNorm: A Principled Approach (...) <https://arxiv.org/abs/2009.03294>`_.
    To this end, the (positive) :obj:`axis` parameter must be strictly > 0 and ideally > 1,
    since first two dimensions are flattened for normalization.

    """

    def __init__(self,
                 axis=-1,
                 epsilon=1e-3, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        r"""Initialize layer :obj:`GraphLayerNormalization`.

        Args:
            axis: Integer or List/Tuple. The axis or axes to normalize across.
                Typically this is the features axis/axes. The left-out axes are
                typically the batch axis/axes. This argument defaults to `-1`, the last dimension in the input.
            epsilon: Small float added to variance to avoid dividing by zero. Defaults to 1e-3.
            center: If True, add offset of `beta` to normalized tensor. If False,
                `beta` is ignored. Defaults to True.
            scale: If True, multiply by `gamma`. If False, `gamma` is not used.
                Defaults to True. When the next layer is linear (also e.g. `nn.relu`),
                this can be disabled since the scaling will be done by the next layer.
            beta_initializer: Initializer for the beta weight. Defaults to zeros.
            gamma_initializer: Initializer for the gamma weight. Defaults to ones.
            beta_regularizer: Optional regularizer for the beta weight. None by default.
            gamma_regularizer: Optional regularizer for the gamma weight. None by default.
            beta_constraint: Optional constraint for the beta weight. None by default.
            gamma_constraint: Optional constraint for the gamma weight. None by default.
        """
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
            inputs (tf.RaggedTensor, tf.Tensor): Embeddings of shape (batch, [M], F, ...)

        Returns:
            tf.RaggedTensor: Normalized ragged tensor of identical shape (batch, [M], F, ...)
        """
        inputs = self.assert_ragged_input_rank(inputs, ragged_rank=1)  # Must have ragged_rank = 1.
        return self.call_on_values_tensor_of_ragged(self._layer_norm, inputs, **kwargs)

    def get_config(self):
        """Get layer configuration."""
        config = super(GraphLayerNormalization, self).get_config()
        config.update({"axis": self.axis})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='GraphBatchNormalization')
class GraphBatchNormalization(GraphBaseLayer):
    r"""Graph batch normalization for (ragged) graph tensor objects.

    Uses `ks.layers.BatchNormalization` on all node or edge features in a batch.
    Following convention suggested by `GraphNorm: A Principled Approach (...) <https://arxiv.org/abs/2009.03294>`_.
    To this end, the (positive) :obj:`axis` parameter must be strictly > 0 and ideally > 1,
    since first two dimensions are flattened for normalization.

    """
    def __init__(self,
                 axis=-1,
                 momentum=0.99, epsilon=0.001, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones', beta_regularizer=None,
                 gamma_regularizer=None, beta_constraint=None, gamma_constraint=None,
                 **kwargs):
        r"""Initialize layer :obj:`GraphBatchNormalization`.

        Args:
            axis: Integer, the axis that should be normalized (typically the features
                axis). For instance, after a `Conv2D` layer with
                `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
            momentum: Momentum for the moving average.
            epsilon: Small float added to variance to avoid dividing by zero.
            center: If True, add offset of `beta` to normalized tensor. If False,
                `beta` is ignored.
            scale: If True, multiply by `gamma`. If False, `gamma` is not used. When
                the next layer is linear (also e.g. `nn.relu`), this can be disabled
                since the scaling will be done by the next layer.
            beta_initializer: Initializer for the beta weight.
            gamma_initializer: Initializer for the gamma weight.
            moving_mean_initializer: Initializer for the moving mean.
            moving_variance_initializer: Initializer for the moving variance.
            beta_regularizer: Optional regularizer for the beta weight.
            gamma_regularizer: Optional regularizer for the gamma weight.
            beta_constraint: Optional constraint for the beta weight.
            gamma_constraint: Optional constraint for the gamma weight.
        """
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
            inputs (tf.RaggedTensor, tf.Tensor): Embeddings of shape (batch, [M], F, ...)

        Returns:
            tf.RaggedTensor: Normalized ragged tensor of identical shape (batch, [M], F, ...)
        """
        inputs = self.assert_ragged_input_rank(inputs, ragged_rank=1)  # Must have ragged_rank = 1.
        return self.call_on_values_tensor_of_ragged(self._layer_norm, inputs, **kwargs)

    def get_config(self):
        """Get layer configuration."""
        config = super(GraphBatchNormalization, self).get_config()
        config.update({"axis": self.axis})
        return config

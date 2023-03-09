import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.axis import get_positive_axis
from kgcnn.ops.segment import segment_ops_by_name

ks = tf.keras

global_normalization_args = {
    "BatchNormalization": [
        "axis", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer", "beta_regularizer",
        "gamma_regularizer", "beta_constraint", "gamma_constraint", "momentum", "moving_mean_initializer",
        "moving_variance_initializer"
    ],
    "GraphBatchNormalization": [
        "axis", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer", "beta_regularizer",
        "gamma_regularizer", "beta_constraint", "gamma_constraint", "momentum", "moving_mean_initializer",
        "moving_variance_initializer"
    ],
    "LayerNormalization": [
        "axis", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer", "beta_regularizer",
        "gamma_regularizer", "beta_constraint", "gamma_constraint"
    ],
    "GraphLayerNormalization": [
        "axis", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer", "beta_regularizer",
        "gamma_regularizer", "beta_constraint", "gamma_constraint"
    ],
    "GraphNormalization": [
        "mean_shift", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer", "alpha_initializer",
        "beta_regularizer", "gamma_regularizer", "beta_constraint", "alpha_constraint", "gamma_constraint",
        "alpha_regularizer"
    ],
    "GraphInstanceNormalization": [
        "epsilon", "center", "scale", "beta_initializer", "gamma_initializer", "alpha_initializer", "beta_regularizer",
        "gamma_regularizer", "beta_constraint", "alpha_constraint", "gamma_constraint", "alpha_regularizer"
    ],
    "GroupNormalization": [
        "groups", "axis", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer", "beta_regularizer",
        "gamma_regularizer", "beta_constraint", "gamma_constraint"]
}


@ks.utils.register_keras_serializable(package='kgcnn', name='GraphLayerNormalization')
class GraphLayerNormalization(GraphBaseLayer):
    r"""Graph Layer normalization for (ragged) graph tensor objects.

    Uses `ks.layers.LayerNormalization` on all node or edge features in a batch.
    Following convention suggested by `GraphNorm: A Principled Approach (...) <https://arxiv.org/abs/2009.03294>`__ .
    To this end, the (positive) :obj:`axis` parameter must be strictly > 0 and ideally > 1,
    since first two dimensions are flattened for normalization.

    The definition of normalization terms for graph neural networks can be categorized as follows. Here we copy the
    definition and description of <https://arxiv.org/abs/2009.03294>`_ . Note that for keras the batch dimension is
    the first dimension.

    .. math::

        \text{Norm}(\hat{h}_{i,j,g}) = \gamma \cdot \frac{\hat{h}_{i,j,g} - \mu}{\sigma} + \beta,


    Consider a batch of graphs :math:`{G_{1}, \dots , G_{b}}` where :math:`b` is the batch size.
    Let :math:`n_{g}` be the number of nodes in graph :math:`G_{g}` .
    We generally denote :math:`\hat{h}_{i,j,g}` as the inputs to the normalization module, e.g.,
    the :math:`j` -th feature value of node :math:`v_i` of graph :math:`G_{g}` ,
    :math:`i = 1, \dots , n_{g}` , :math:`j = 1, \dots , d` , :math:`g = 1, \dots , b` .

    To adapt Layer-Norm to GNNs, we view each node as a basic component, resembling words in a sentence, and apply
    normalization to all feature values across different dimensions of each node,
    i.e. , over dimension :math:`j` of :math:`\hat{h}_{i,j,g}` .

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
        return self.map_values(self._layer_norm, inputs, **kwargs)

    def get_config(self):
        """Get layer configuration."""
        config = super(GraphLayerNormalization, self).get_config()
        config.update({"axis": self.axis})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='GraphBatchNormalization')
class GraphBatchNormalization(GraphBaseLayer):
    r"""Graph batch normalization for (ragged) graph tensor objects.

    Uses `ks.layers.BatchNormalization` on all node or edge features in a batch.
    Following convention suggested by `GraphNorm: A Principled Approach (...) <https://arxiv.org/abs/2009.03294>`__ .
    To this end, the (positive) :obj:`axis` parameter must be strictly > 0 and ideally > 1,
    since first two dimensions are flattened for normalization.

    The definition of normalization terms for graph neural networks can be categorized as follows. Here we copy the
    definition and description of `<https://arxiv.org/abs/2009.03294>`_ . Note that for keras the batch dimension is
    the first dimension.

    .. math::

        \text{Norm}(\hat{h}_{i,j,g}) = \gamma \cdot \frac{\hat{h}_{i,j,g} - \mu}{\sigma} + \beta,


    Consider a batch of graphs :math:`{G_{1}, \dots , G_{b}}` where :math:`b` is the batch size.
    Let :math:`n_{g}` be the number of nodes in graph :math:`G_{g}` .
    We generally denote :math:`\hat{h}_{i,j,g}` as the inputs to the normalization module, e.g.,
    the :math:`j` -th feature value of node :math:`v_i` of graph :math:`G_{g}` ,
    :math:`i = 1, \dots , n_{g}` , :math:`j = 1, \dots , d` , :math:`g = 1, \dots , b` .

    For BatchNorm, normalization and the computation of :math:`mu`
    and :math:`\sigma` are applied to all values in the same feature dimension
    across the nodes of all graphs in the batch as in
    `Xu et al. (2019) <https://openreview.net/forum?id=ryGs6iA5Km>`__ , i.e., over dimensions :math:`g`, :math:`i`
    of :math:`\hat{h}_{i,j,g}` .

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
                                                       "beta_regularizer", "gamma_regularizer", "beta_constraint",
                            "gamma_constraint"]}

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
        return self.map_values(self._layer_norm, inputs, **kwargs)

    def get_config(self):
        """Get layer configuration."""
        config = super(GraphBatchNormalization, self).get_config()
        config.update({"axis": self.axis})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='GraphNormalization')
class GraphNormalization(GraphBaseLayer):
    r"""Graph normalization for (ragged) graph tensor objects.

    Following convention suggested by `GraphNorm: A Principled Approach (...) <https://arxiv.org/abs/2009.03294>`__ .

    The definition of normalization terms for graph neural networks can be categorized as follows. Here we copy the
    definition and description of `<https://arxiv.org/abs/2009.03294>`_ . Note that for keras the batch dimension is
    the first dimension.

    .. math::

        \text{Norm}(\hat{h}_{i,j,g}) = \gamma \cdot \frac{\hat{h}_{i,j,g} - \mu}{\sigma} + \beta,


    Consider a batch of graphs :math:`{G_{1}, \dots , G_{b}}` where :math:`b` is the batch size.
    Let :math:`n_{g}` be the number of nodes in graph :math:`G_{g}` .
    We generally denote :math:`\hat{h}_{i,j,g}` as the inputs to the normalization module, e.g.,
    the :math:`j` -th feature value of node :math:`v_i` of graph :math:`G_{g}` ,
    :math:`i = 1, \dots , n_{g}` , :math:`j = 1, \dots , d` , :math:`g = 1, \dots , b` .

    For InstanceNorm, we regard each graph as an instance. The normalization is
    then applied to the feature values across all nodes for each
    individual graph, i.e., over dimension :math:`i` of :math:`\hat{h}_{i,j,g}` .

    Additionally, the following proposed additions for GraphNorm are added when compared to InstanceNorm.

    .. math::

        \text{GraphNorm}(\hat{h}_{i,j}) = \gamma_j \cdot \frac{\hat{h}_{i,j} - \alpha_j \mu_j }{\hat{\sigma}_j}+\beta_j

    where :math:`\mu_j = \frac{\sum^n_{i=1} hat{h}_{i,j}}{n}` ,
    :math:`\hat{\sigma}^2_j = \frac{\sum^n_{i=1} (hat{h}_{i,j} - \alpha_j \mu_j)^2}{n}`  ,
    and :math:`\gamma_j` , :math:`beta_j` are the affine parameters as in other normalization methods.

    .. code-block:: python

        import tensorflow as tf
        from kgcnn.layers.norm import GraphNormalization
        layer = GraphNormalization()
        test = tf.ragged.constant([[[0.0, 0.0],[1.0, -1.0]],[[1.0, 1.0],[0.0, 0.0],[-2.0, -2.0]]], ragged_rank=1)
        print(layer(test))

    """

    def __init__(self,
                 mean_shift=True, epsilon=1e-3, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones', alpha_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, alpha_regularizer=None,
                 beta_constraint=None, gamma_constraint=None, alpha_constraint=None,
                 **kwargs):
        r"""Initialize layer :obj:`GraphBatchNormalization`.

        Args:
            epsilon: Small float added to variance to avoid dividing by zero. Defaults to 1e-3.
            center: If True, add offset of `beta` to normalized tensor. If False,
                `beta` is ignored. Defaults to True.
            scale: If True, multiply by `gamma`. If False, `gamma` is not used.
                Defaults to True. When the next layer is linear (also e.g. `nn.relu`),
                this can be disabled since the scaling will be done by the next layer.
            mean_shift (bool): Whether to apply alpha. Default is True.
            beta_initializer: Initializer for the beta weight. Defaults to 'zeros'.
            gamma_initializer: Initializer for the gamma weight. Defaults to 'ones'.
            alpha_initializer: Initializer for the alpha weight. Defaults to 'ones'.
            beta_regularizer: Optional regularizer for the beta weight. None by default.
            gamma_regularizer: Optional regularizer for the gamma weight. None by default.
            alpha_regularizer: Optional regularizer for the alpha weight. None by default.
            beta_constraint: Optional constraint for the beta weight. None by default.
            gamma_constraint: Optional constraint for the gamma weight. None by default.
            alpha_constraint: Optional constraint for the alpha weight. None by default.
        """
        super(GraphNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        self._eps = tf.constant(epsilon, dtype=self.dtype)
        self.center = center
        self.mean_shift = mean_shift
        self.scale = scale
        self.beta_initializer = ks.initializers.get(beta_initializer)
        self.gamma_initializer = ks.initializers.get(gamma_initializer)
        self.alpha_initializer = ks.initializers.get(alpha_initializer)
        self.beta_regularizer = ks.regularizers.get(beta_regularizer)
        self.gamma_regularizer = ks.regularizers.get(gamma_regularizer)
        self.alpha_regularizer = ks.regularizers.get(alpha_regularizer)
        self.beta_constraint = ks.constraints.get(beta_constraint)
        self.gamma_constraint = ks.constraints.get(gamma_constraint)
        self.alpha_constraint = ks.constraints.get(alpha_constraint)
        # Weights
        self.alpha = None
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        """Build layer."""
        super(GraphNormalization, self).build(input_shape)
        param_shape = [x if x is not None else 1 for x in input_shape[2:]]
        if self.scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False,
            )

        if self.center:
            self.beta = self.add_weight(
                name="beta",
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False,
            )

        if self.mean_shift:
            self.alpha = self.add_weight(
                name="alpha",
                shape=param_shape,
                initializer=self.alpha_initializer,
                regularizer=self.alpha_regularizer,
                constraint=self.alpha_constraint,
                trainable=True,
                experimental_autocast=False,
            )

        self.built = True

    def _ragged_mean_std(self, inputs):
        # Here a segment operation for ragged_rank=1 tensors is used.
        # Alternative is to simply use tf.reduce_mean which should also work for latest tf-version.
        # Then tf.nn.moments could be used or similar tf implementation for variance and mean.
        values = inputs.values
        if values.dtype in ("float16", "bfloat16") and self.dtype == "float32":
            values = tf.cast(values, "float32")
        mean = segment_ops_by_name("mean", values, inputs.value_rowids())
        if self.mean_shift:
            mean = mean * tf.expand_dims(self.alpha, axis=0)
        mean = tf.repeat(mean, inputs.row_lengths(), axis=0)
        diff = values - mean
        # Not sure whether to stop gradients for variance if alpha ist used.
        square_diff = tf.square(diff)  # values - tf.stop_gradient(mean)
        variance = segment_ops_by_name("mean", square_diff, inputs.value_rowids())
        std = tf.sqrt(variance + self._eps)
        std = tf.repeat(std, inputs.row_lengths(), axis=0)
        return mean, std, diff / std

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.RaggedTensor, tf.Tensor): Node or edge embeddings of shape (batch, [M], F, ...)

        Returns:
            tf.RaggedTensor: Normalized ragged tensor of identical shape (batch, [M], F, ...)
        """
        inputs = self.assert_ragged_input_rank(inputs, ragged_rank=1)  # Must have ragged_rank = 1.
        mean, std, new_values = self._ragged_mean_std(inputs)
        # Recomputing diff.
        if self.scale:
            new_values = new_values * tf.expand_dims(self.gamma, axis=0)
        if self.center:
            new_values = new_values + self.beta
        return inputs.with_values(new_values)

    def get_config(self):
        """Get layer configuration."""
        config = super(GraphNormalization, self).get_config()
        config.update({
            "mean_shift": self.mean_shift,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": ks.initializers.serialize(self.beta_initializer),
            "gamma_initializer": ks.initializers.serialize(self.gamma_initializer),
            "alpha_initializer": ks.initializers.serialize(self.alpha_initializer),
            "beta_regularizer": ks.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": ks.regularizers.serialize(self.gamma_regularizer),
            "alpha_regularizer": ks.regularizers.serialize(self.alpha_regularizer),
            "beta_constraint": ks.constraints.serialize(self.beta_constraint),
            "gamma_constraint": ks.constraints.serialize(self.gamma_constraint),
            "alpha_constraint": ks.constraints.serialize(self.alpha_constraint),
        })
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='GraphInstanceNormalization')
class GraphInstanceNormalization(GraphNormalization):
    r"""Graph instance normalization for (ragged) graph tensor objects.

    Following convention suggested by `GraphNorm: A Principled Approach (...) <https://arxiv.org/abs/2009.03294>`__ .

    The definition of normalization terms for graph neural networks can be categorized as follows. Here we copy the
    definition and description of `<https://arxiv.org/abs/2009.03294>`_ . Note that for keras the batch dimension is
    the first dimension.

    .. math::

        \text{Norm}(\hat{h}_{i,j,g}) = \gamma \cdot \frac{\hat{h}_{i,j,g} - \mu}{\sigma} + \beta,

    Consider a batch of graphs :math:`{G_{1}, \dots , G_{b}}` where :math:`b` is the batch size.
    Let :math:`n_{g}` be the number of nodes in graph :math:`G_{g}` .
    We generally denote :math:`\hat{h}_{i,j,g}` as the inputs to the normalization module, e.g.,
    the :math:`j` -th feature value of node :math:`v_i` of graph :math:`G_{g}` ,
    :math:`i = 1, \dots , n_{g}` , :math:`j = 1, \dots , d` , :math:`g = 1, \dots , b` .

    For InstanceNorm, we regard each graph as an instance. The normalization is
    then applied to the feature values across all nodes for each
    individual graph, i.e., over dimension :math:`i` of :math:`\hat{h}_{i,j,g}` .

    .. code-block:: python

        import tensorflow as tf
        from kgcnn.layers.norm import GraphInstanceNormalization
        layer = GraphInstanceNormalization()
        test = tf.ragged.constant([[[0.0, 0.0],[1.0, -1.0]],[[1.0, 1.0],[0.0, 0.0],[-2.0, -2.0]]], ragged_rank=1)
        print(layer(test))

    """

    def __init__(self, **kwargs):
        r"""Initialize layer :obj:`GraphBatchNormalization`.

        Args:
            epsilon: Small float added to variance to avoid dividing by zero. Defaults to 1e-3.
            center: If True, add offset of `beta` to normalized tensor. If False,
                `beta` is ignored. Defaults to True.
            scale: If True, multiply by `gamma`. If False, `gamma` is not used.
                Defaults to True. When the next layer is linear (also e.g. `nn.relu`),
                this can be disabled since the scaling will be done by the next layer.
            beta_initializer: Initializer for the beta weight. Defaults to 'zeros'.
            gamma_initializer: Initializer for the gamma weight. Defaults to 'ones'.
            alpha_initializer: Initializer for the alpha weight. Defaults to 'ones'.
            beta_regularizer: Optional regularizer for the beta weight. None by default.
            gamma_regularizer: Optional regularizer for the gamma weight. None by default.
            alpha_regularizer: Optional regularizer for the alpha weight. None by default.
            beta_constraint: Optional constraint for the beta weight. None by default.
            gamma_constraint: Optional constraint for the gamma weight. None by default.
            alpha_constraint: Optional constraint for the alpha weight. None by default.

        """
        super(GraphInstanceNormalization, self).__init__(mean_shift=False, **kwargs)

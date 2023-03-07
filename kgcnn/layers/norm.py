import tensorflow as tf
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.axis import get_positive_axis
ks = tf.keras


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
        return self.map_values(self._layer_norm, inputs, **kwargs)

    def get_config(self):
        """Get layer configuration."""
        config = super(GraphBatchNormalization, self).get_config()
        config.update({"axis": self.axis})
        return config


@ks.utils.register_keras_serializable(package='kgcnn', name='GraphInstanceNormalization')
class GraphInstanceNormalization(GraphBaseLayer):
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

    """

    def __init__(self,
                 axis=None,
                 epsilon=1e-3, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        r"""Initialize layer :obj:`GraphBatchNormalization`.

        Args:
            axis: Integer or List/Tuple. The axis or axes to normalize across in addition to graph instances.
                This should be always > 1 or None. Default is None.
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
        super(GraphInstanceNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(GraphInstanceNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.RaggedTensor, tf.Tensor): Node or edge embeddings of shape (batch, [M], F, ...)

        Returns:
            tf.RaggedTensor: Normalized ragged tensor of identical shape (batch, [M], F, ...)
        """
        raise NotImplementedError("Not yet implemented")

    def get_config(self):
        """Get layer configuration."""
        config = super(GraphInstanceNormalization, self).get_config()
        config.update({})
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

    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        r"""Initialize layer :obj:`GraphBatchNormalization`.

        Args:
            axis: Integer or List/Tuple. The axis or axes to normalize across in addition to graph instances.
                This should be always > 1 or None. Default is None.
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
        super(GraphNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(GraphNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (tf.RaggedTensor, tf.Tensor): Node or edge embeddings of shape (batch, [M], F, ...)

        Returns:
            tf.RaggedTensor: Normalized ragged tensor of identical shape (batch, [M], F, ...)
        """
        raise NotImplementedError("Not yet implemented")

    def get_config(self):
        """Get layer configuration."""
        config = super(GraphNormalization, self).get_config()
        config.update({})
        return config



import keras as ks
from keras.layers import Layer
from keras import ops
from keras import InputSpec
from kgcnn.ops.scatter import scatter_reduce_sum
from keras.layers import LayerNormalization
from keras.layers import BatchNormalization

global_normalization_args = {
    "GraphNormalization": (
        "mean_shift", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer", "alpha_initializer",
        "beta_regularizer", "gamma_regularizer", "beta_constraint", "alpha_constraint", "gamma_constraint",
        "alpha_regularizer"
    ),
    "GraphInstanceNormalization": (
        "epsilon", "center", "scale", "beta_initializer", "gamma_initializer", "alpha_initializer", "beta_regularizer",
        "gamma_regularizer", "beta_constraint", "alpha_constraint", "gamma_constraint", "alpha_regularizer"
    ),
    "GraphBatchNormalization": (
        "axis", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer", "beta_regularizer",
        "gamma_regularizer", "beta_constraint", "gamma_constraint", "momentum", "moving_mean_initializer",
        "moving_variance_initializer", "padded_disjoint"
    ),
    "GraphLayerNormalization": (
        "axis", "epsilon", "center", "scale", "beta_initializer", "gamma_initializer", "beta_regularizer",
        "gamma_regularizer", "beta_constraint", "gamma_constraint"
    ),
}


class GraphLayerNormalization(LayerNormalization):

    def __init__(self, **kwargs):
        super(GraphLayerNormalization, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return super(GraphLayerNormalization, self).compute_output_shape(input_shape[0])

    def build(self, input_shape):
        super(GraphLayerNormalization, self).build(input_shape[0])

    def call(self, inputs, **kwargs):
        return super(GraphLayerNormalization, self).call(inputs[0])

    def get_config(self):
        return super(GraphLayerNormalization, self).get_config()


class GraphBatchNormalization(BatchNormalization):

    def __init__(self, padded_disjoint: bool = False, **kwargs):
        super(GraphBatchNormalization, self).__init__(**kwargs)
        self.padded_disjoint = padded_disjoint

    def compute_output_shape(self, input_shape):
        return super(GraphBatchNormalization, self).compute_output_shape(input_shape[0])

    def build(self, input_shape):
        super(GraphBatchNormalization, self).build(input_shape[0])
        self.input_spec = [
            InputSpec(ndim=len(input_shape[0]), axes={self.axis: input_shape[0][self.axis]}),
            InputSpec(ndim=len(input_shape[1])),
            InputSpec(ndim=len(input_shape[2])),
        ]

    def call(self, inputs, training=None, **kwargs):
        if not self.padded_disjoint:
            return super(GraphBatchNormalization, self).call(inputs[0], training=training)
        else:
            padded_mask = inputs[1] > 0
            return super(GraphBatchNormalization, self).call(inputs[0], training=training, mask=padded_mask)

    def get_config(self):
        config = super(GraphBatchNormalization, self).get_config()
        config.update({"padded_disjoint": self.padded_disjoint})
        return config


class GraphNormalization(Layer):
    r"""Graph normalization for graph tensor objects.

    Following convention suggested by `GraphNorm: A Principled Approach (...) <https://arxiv.org/abs/2009.03294>`__ .

    The definition of normalization terms for graph neural networks can be categorized as follows. Here we copy the
    definition and description of `<https://arxiv.org/abs/2009.03294>`_ .

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

    where :math:`\mu_j = \frac{\sum^n_{i=1} \hat{h}_{i,j}}{n}` ,
    :math:`\hat{\sigma}^2_j = \frac{\sum^n_{i=1} (\hat{h}_{i,j} - \alpha_j \mu_j)^2}{n}`  ,
    and :math:`\gamma_j` , :math:`beta_j` are the affine parameters as in other normalization methods.

    .. code-block:: python

        from kgcnn.layers.norm import GraphNormalization
        layer = GraphNormalization()
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
        self._eps = ops.convert_to_tensor(epsilon, dtype=self.dtype)
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
        param_shape = [x if x is not None else 1 for x in input_shape[0]]
        if self.scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
            )

        if self.center:
            self.beta = self.add_weight(
                name="beta",
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
            )

        if self.mean_shift:
            self.alpha = self.add_weight(
                name="alpha",
                shape=param_shape,
                initializer=self.alpha_initializer,
                regularizer=self.alpha_regularizer,
                constraint=self.alpha_constraint,
                trainable=True,
            )

        self.built = True

    def _ragged_mean_std(self, inputs: list):
        values, row_ids, lengths = inputs

        if values.dtype in ("float16", "bfloat16") and self.dtype == "float32":
            values = ops.cast(values, "float32")

        shape_ = ops.shape(lengths)[:1] + ops.shape(values)[1:]
        counts_ = scatter_reduce_sum(row_ids, ops.ones_like(values), shape=shape_)

        mean = scatter_reduce_sum(row_ids, values, shape=shape_)/counts_
        if self.mean_shift:
            mean = mean * self.alpha
        mean = ops.take(mean, row_ids, axis=0)
        diff = values - mean
        # Not sure whether to stop gradients for variance if alpha ist used.
        square_diff = ops.square(diff)  # values - tf.stop_gradient(mean)
        variance = scatter_reduce_sum(row_ids, square_diff, shape=shape_)/counts_
        std = ops.sqrt(variance + self._eps)
        std = ops.take(std, row_ids, axis=0)
        return mean, std, diff / std

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): `[values, graph_id, reference]` .

                - values (Tensor): Tensor to normalize of shape `(None, F, ...)` .
                - graph_id (Tensor): Tensor of graph IDs of shape `(None, )` .
                - reference (Tensor, optional): Graph reference of disjoint batch of shape `(batch, )` .

        Returns:
            Tensor: Normalized tensor of identical shape (None, F, ...)
        """
        mean, std, new_values = self._ragged_mean_std(inputs)
        # Recomputing diff.
        if self.scale:
            new_values = new_values * self.gamma
        if self.center:
            new_values = new_values + self.beta
        return new_values

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


class GraphInstanceNormalization(GraphNormalization):
    r"""Graph instance normalization for graph tensor objects.

    Following convention suggested by `GraphNorm: A Principled Approach (...) <https://arxiv.org/abs/2009.03294>`__ .

    The definition of normalization terms for graph neural networks can be categorized as follows. Here we copy the
    definition and description of `<https://arxiv.org/abs/2009.03294>`_ .

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

        from kgcnn.layers.norm import GraphInstanceNormalization
        layer = GraphInstanceNormalization()
    """

    def __init__(self, **kwargs):
        r"""Initialize layer :obj:`GraphBatchNormalization` .

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

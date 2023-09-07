from keras_core.layers import Layer, Dense, Activation, Add, Multiply
from kgcnn.layers_core.aggr import AggregateWeightedLocalEdges, AggregateLocalEdges
from kgcnn.layers_core.gather import GatherNodesOutgoing
from keras_core import ops


class GCN(Layer):
    r"""Graph convolution according to `Kipf et al <https://arxiv.org/abs/1609.02907>`__ .

    Computes graph convolution as :math:`\sigma(A_s(WX+b))` where :math:`A_s` is the precomputed and scaled adjacency
    matrix. The scaled adjacency matrix is defined by :math:`A_s = D^{-0.5} (A + I) D^{-0.5}` with the degree
    matrix :math:`D` . In place of :math:`A_s` , this layers uses edge features (that are the entries of :math:`A_s` )
    and edge indices.

    .. note::

        :math:`A_s` is considered pre-scaled, this is not done by this layer!
        If no scaled edge features are available, you could consider use e.g. "mean",
        or :obj:`normalize_by_weights` to obtain a similar behaviour that is expected b
        y a pre-scaled adjacency matrix input.

    Edge features must be possible to broadcast to node features, since they are multiplied with the node features.
    Ideally they are weights of shape `(..., 1)` for broadcasting, e.g. entries of :math:`A_s` .

    Args:
        units (int): Output dimension/ units of dense layer.
        pooling_method (str): Pooling method for summing edges. Default is 'segment_sum'.
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
            In this case the edge features are considered weights of dimension (...,1) and are summed for each node.
        activation (str): Activation. Default is 'kgcnn>leaky_relu'.
        use_bias (bool): Use bias. Default is True.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self,
                 units,
                 pooling_method='scatter_sum',
                 normalize_by_weights=False,
                 activation='kgcnn>leaky_relu',
                 use_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize layer."""
        super(GCN, self).__init__(**kwargs)
        self.normalize_by_weights = normalize_by_weights
        self.pooling_method = pooling_method
        self.units = units
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer, "use_bias": use_bias}
        pool_args = {"pooling_method": pooling_method, "normalize_by_weights": normalize_by_weights}

        # Layers
        self.layer_gather = GatherNodesOutgoing()
        self.layer_dense = Dense(units=self.units, activation='linear', **kernel_args)
        self.layer_pool = AggregateWeightedLocalEdges(**pool_args)
        self.layer_act = Activation(activation)

    def build(self, input_shape):
        assert isinstance(input_shape, list), "Require list input"
        self.layer_dense.build(input_shape[0])
        dense_shape = self.layer_dense.compute_output_shape(input_shape[0])
        self.layer_gather.build([dense_shape, input_shape[2]])
        gather_shape = self.layer_gather.compute_output_shape([dense_shape, input_shape[2]])
        self.layer_pool.build([input_shape[0], gather_shape, input_shape[2], input_shape[1]])
        pool_shape = self.layer_pool.compute_output_shape(
            [input_shape[0], gather_shape, input_shape[2], input_shape[1]])
        self.layer_act.build(pool_shape)
        self.built = True

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, edge_index]

                - nodes (Tensor): Node embeddings of shape `(None, F)`
                - edges (Tensor): Edge or message embeddings of shape `(None, F)`
                - edge_index (Tensor): Edge indices referring to nodes of shape `(2, None)`

        Returns:
            Tensor: Node embeddings of shape `(None, F)`
        """
        node, edges, edge_index = inputs
        no = self.layer_dense(node, **kwargs)
        no = self.layer_gather([no, edge_index], **kwargs)
        nu = self.layer_pool([node, no, edge_index, edges], **kwargs)  # Summing for each node connection
        out = self.layer_act(nu, **kwargs)
        return out

    def get_config(self):
        """Update config."""
        config = super(GCN, self).get_config()
        config.update({"normalize_by_weights": self.normalize_by_weights,
                       "pooling_method": self.pooling_method, "units": self.units})
        conf_dense = self.layer_dense.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "use_bias"]:
            if x in conf_dense:
                config.update({x: conf_dense[x]})
        conf_act = self.layer_act.get_config()
        config.update({"activation": conf_act["activation"]})
        return config


class SchNetCFconv(Layer):
    r"""Continuous filter convolution of `SchNet <https://aip.scitation.org/doi/pdf/10.1063/1.5019779>`__ .

    Edges are processed by 2 :obj:`Dense` layers, multiplied on outgoing node features and pooled for receiving node.

    Args:
        units (int): Units for Dense layer.
        cfconv_pool (str): Pooling method. Default is 'segment_sum'.
        use_bias (bool): Use bias. Default is True.
        activation (str): Activation function. Default is 'kgcnn>shifted_softplus'.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self, units,
                 cfconv_pool='scatter_sum',
                 use_bias=True,
                 activation='shifted_softplus',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize Layer."""
        super(SchNetCFconv, self).__init__(**kwargs)
        self.cfconv_pool = cfconv_pool
        self.units = units
        self.use_bias = use_bias
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}
        # Layer
        self.lay_dense1 = Dense(units=self.units, activation=activation, use_bias=self.use_bias, **kernel_args)
        self.lay_dense2 = Dense(units=self.units, activation='linear', use_bias=self.use_bias, **kernel_args)
        self.lay_sum = AggregateLocalEdges(pooling_method=cfconv_pool)
        self.gather_n = GatherNodesOutgoing()
        self.lay_mult = Multiply()

    def build(self, input_shape):
        """Build layer."""
        super(SchNetCFconv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass. Calculate edge update.

        Args:
            inputs: [nodes, edges, edge_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [N], F)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [N], 2)

        Returns:
            tf.RaggedTensor: Updated node features.
        """
        node, edge, disjoint_indices = inputs
        x = self.lay_dense1(edge, **kwargs)
        x = self.lay_dense2(x, **kwargs)
        node2exp = self.gather_n([node, disjoint_indices], **kwargs)
        x = self.lay_mult([node2exp, x], **kwargs)
        x = self.lay_sum([node, x, disjoint_indices], **kwargs)
        return x

    def get_config(self):
        """Update layer config."""
        config = super(SchNetCFconv, self).get_config()
        config.update({"cfconv_pool": self.cfconv_pool, "units": self.units})
        config_dense = self.lay_dense1.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias"]:
            config.update({x: config_dense[x]})
        return config


class SchNetInteraction(Layer):
    r"""`SchNet <https://aip.scitation.org/doi/pdf/10.1063/1.5019779>`__ interaction block,
    which uses the continuous filter convolution from :obj:`SchNetCFconv`.

    Args:
        units (int): Dimension of node embedding. Default is 128.
        cfconv_pool (str): Pooling method information for SchNetCFconv layer. Default is'segment_sum'.
        use_bias (bool): Use bias in last layers. Default is True.
        activation (str): Activation function. Default is 'kgcnn>shifted_softplus'.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
    """

    def __init__(self,
                 units=128,
                 cfconv_pool='scatter_sum',
                 use_bias=True,
                 activation='shifted_softplus',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize Layer."""
        super(SchNetInteraction, self).__init__(**kwargs)
        self.cfconv_pool = cfconv_pool
        self.use_bias = use_bias
        self.units = units
        kernel_args = {"kernel_regularizer": kernel_regularizer, "activity_regularizer": activity_regularizer,
                       "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                       "bias_constraint": bias_constraint, "kernel_initializer": kernel_initializer,
                       "bias_initializer": bias_initializer}
        conv_args = {"units": self.units, "use_bias": use_bias, "activation": activation, "cfconv_pool": cfconv_pool}

        # Layers
        self.lay_cfconv = SchNetCFconv(**conv_args, **kernel_args)
        self.lay_dense1 = Dense(units=self.units, activation='linear', use_bias=False, **kernel_args)
        self.lay_dense2 = Dense(units=self.units, activation=activation, use_bias=self.use_bias, **kernel_args)
        self.lay_dense3 = Dense(units=self.units, activation='linear', use_bias=self.use_bias, **kernel_args)
        self.lay_add = Add()

    def build(self, input_shape):
        """Build layer."""
        super(SchNetInteraction, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass. Calculate node update.

        Args:
            inputs: [nodes, edges, tensor_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [N], F)
                - tensor_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [N], 2)

        Returns:
            tf.RaggedTensor: Updated node embeddings of shape (batch, [N], F).
        """
        node, edge, indexlist = inputs
        x = self.lay_dense1(node, **kwargs)
        x = self.lay_cfconv([x, edge, indexlist], **kwargs)
        x = self.lay_dense2(x, **kwargs)
        x = self.lay_dense3(x, **kwargs)
        out = self.lay_add([node, x], **kwargs)
        return out

    def get_config(self):
        config = super(SchNetInteraction, self).get_config()
        config.update({"cfconv_pool": self.cfconv_pool, "units": self.units, "use_bias": self.use_bias})
        conf_dense = self.lay_dense2.get_config()
        for x in ["activation", "kernel_regularizer", "bias_regularizer", "activity_regularizer",
                  "kernel_constraint", "bias_constraint", "kernel_initializer", "bias_initializer"]:
            config.update({x: conf_dense[x]})
        return config
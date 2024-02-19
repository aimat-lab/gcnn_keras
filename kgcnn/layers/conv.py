from keras.layers import Layer, Dense, Activation, Add, Multiply
from kgcnn.layers.aggr import AggregateWeightedLocalEdges, AggregateLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing
from keras import ops
import kgcnn.ops.activ


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
    """

    def __init__(self,
                 units,
                 pooling_method='scatter_sum',
                 normalize_by_weights=False,
                 activation="kgcnn>leaky_relu2",
                 use_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize layer.

        Args:
            units (int): Output dimension/ units of dense layer.
            pooling_method (str): Pooling method for summing edges. Default is 'segment_sum'.
            normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
                In this case the edge features are considered weights of dimension (...,1) and are summed for each node.
            activation (str): Activation. Default is "kgcnn>leaky_relu2".
            use_bias (bool): Use bias. Default is True.
            kernel_regularizer: Kernel regularization. Default is None.
            bias_regularizer: Bias regularization. Default is None.
            activity_regularizer: Activity regularization. Default is None.
            kernel_constraint: Kernel constrains. Default is None.
            bias_constraint: Bias constrains. Default is None.
            kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
            bias_initializer: Initializer for bias. Default is 'zeros'.
        """
        super(GCN, self).__init__(**kwargs)
        # Changes in keras serialization behaviour for activations in 3.0.2.
        # Keep string at least for default. Also renames to prevent clashes with keras leaky_relu.
        if activation in ["kgcnn>leaky_relu", "kgcnn>leaky_relu2"]:
            activation = {"class_name": "function", "config": "kgcnn>leaky_relu2"}
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
        super(GCN, self).build(input_shape)

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
    """

    def __init__(self, units,
                 cfconv_pool="scatter_sum",
                 use_bias=True,
                 activation="kgcnn>shifted_softplus",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize Layer.

        Args:
            units (int): Units for Dense layer.
            cfconv_pool (str): Pooling method. Default is 'segment_sum'.
            use_bias (bool): Use bias. Default is True.
            activation (str): Activation function. Default is "kgcnn>shifted_softplus".
            kernel_regularizer: Kernel regularization. Default is None.
            bias_regularizer: Bias regularization. Default is None.
            activity_regularizer: Activity regularization. Default is None.
            kernel_constraint: Kernel constrains. Default is None.
            bias_constraint: Bias constrains. Default is None.
            kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
            bias_initializer: Initializer for bias. Default is 'zeros'.
        """
        super(SchNetCFconv, self).__init__(**kwargs)
        # Changes in keras serialization behaviour for activations in 3.0.2.
        # Keep string at least for default.
        if activation in ["kgcnn>shifted_softplus"]:
            activation = {"class_name": "function", "config": "kgcnn>shifted_softplus"}
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
        super(SchNetCFconv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass. Calculate edge update.

        Args:
            inputs: [nodes, edges, edge_index]

                - nodes (Tensor): Node embeddings of shape ([N], F)
                - edges (Tensor): Edge or message embeddings of shape ([M], F)
                - edge_index (Tensor): Edge indices referring to nodes of shape (2, [M])

        Returns:
            Tensor: Updated node features.
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
            if x in config_dense:
                config.update({x: config_dense[x]})
        return config


class SchNetInteraction(Layer):
    r"""`SchNet <https://aip.scitation.org/doi/pdf/10.1063/1.5019779>`__ interaction block,
    which uses the continuous filter convolution from :obj:`SchNetCFconv` .
    """

    def __init__(self,
                 units=128,
                 cfconv_pool='scatter_sum',
                 use_bias=True,
                 activation="kgcnn>shifted_softplus",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        """Initialize Layer.

        Args:
            units (int): Dimension of node embedding. Default is 128.
            cfconv_pool (str): Pooling method information for SchNetCFconv layer. Default is 'scatter_sum'.
            use_bias (bool): Use bias in last layers. Default is True.
            activation (str): Activation function. Default is "kgcnn>shifted_softplus".
            kernel_regularizer: Kernel regularization. Default is None.
            bias_regularizer: Bias regularization. Default is None.
            activity_regularizer: Activity regularization. Default is None.
            kernel_constraint: Kernel constrains. Default is None.
            bias_constraint: Bias constrains. Default is None.
            kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
            bias_initializer: Initializer for bias. Default is 'zeros'.
        """
        super(SchNetInteraction, self).__init__(**kwargs)
        # Changes in keras serialization behaviour for activations in 3.0.2.
        # Keep string at least for default.
        if activation in ["kgcnn>shifted_softplus"]:
            activation = {"class_name": "function", "config": "kgcnn>shifted_softplus"}
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
        super(SchNetInteraction, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass. Calculate node update.

        Args:
            inputs: [nodes, edges, tensor_index]

                - nodes (Tensor): Node embeddings of shape ([N], F)
                - edges (Tensor): Edge or message embeddings of shape ([M], F)
                - tensor_index (Tensor): Edge indices referring to nodes of shape (2, [M])

        Returns:
            Tensor: Updated node embeddings of shape ([N], F).
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
            if x in conf_dense:
                config.update({x: conf_dense[x]})
        return config


class GIN(Layer):
    r"""Convolutional unit of `Graph Isomorphism Network from: How Powerful are Graph Neural Networks?
    <https://arxiv.org/abs/1810.00826>`__ .

    Computes graph convolution at step :math:`k` for node embeddings :math:`h_\nu` as:

    .. math::

        h_\nu^{(k)} = \phi^{(k)} ((1+\epsilon^{(k)}) h_\nu^{k-1} + \sum_{u\in N(\nu)}) h_u^{k-1}.

    with optional learnable :math:`\epsilon^{(k)}`

    .. note::

        The non-linear mapping :math:`\phi^{(k)}` , usually an :obj:`MLP` , is not included in this layer.
    """

    def __init__(self,
                 pooling_method='scatter_sum',
                 epsilon_learnable=False,
                 **kwargs):
        """Initialize layer.

        Args:
            epsilon_learnable (bool): If epsilon is learnable or just constant zero. Default is False.
            pooling_method (str): Pooling method for summing edges. Default is 'segment_sum'.
        """
        super(GIN, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.epsilon_learnable = epsilon_learnable

        # Layers
        self.lay_gather = GatherNodesOutgoing()
        self.lay_pool = AggregateLocalEdges(pooling_method=self.pooling_method)
        self.lay_add = Add()

        # Epsilon with trainable as optional and default zeros initialized.
        self.eps_k = self.add_weight(shape=tuple(), name="epsilon_k", trainable=self.epsilon_learnable,
                                     initializer="zeros", dtype=self.dtype)

    def build(self, input_shape):
        """Build layer."""
        super(GIN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [nodes, edge_index]

                - nodes (Tensor): Node embeddings of shape `([N], F)`
                - edge_index (Tensor): Edge indices referring to nodes of shape `(2, [M])`

        Returns:
            Tensor: Node embeddings of shape `([N], F)`
        """
        node, edge_index = inputs
        ed = self.lay_gather([node, edge_index], **kwargs)
        nu = self.lay_pool([node, ed, edge_index], **kwargs)  # Summing for each node connection
        no = (ops.convert_to_tensor(1, dtype=self.eps_k.dtype) + self.eps_k) * node
        out = self.lay_add([no, nu], **kwargs)
        return out

    def get_config(self):
        """Update config."""
        config = super(GIN, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "epsilon_learnable": self.epsilon_learnable})
        return config


class GINE(Layer):
    r"""Convolutional unit of `Strategies for Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>`__ .

    Computes graph convolution with node embeddings :math:`\mathbf{h}` and compared to :obj:`GIN_conv`,
    adds edge embeddings of :math:`\mathbf{e}_{ij}`.

    .. math::

        \mathbf{h}^{\prime}_i = f_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{h}_i + \sum_{j \in \mathcal{N}(i)} \phi \; ( \mathbf{h}_j + \mathbf{e}_{ij} ) \right),

    with optionally learnable :math:`\epsilon`. The activation :math:`\phi` can be chosen differently
    but defaults to RELU.

    .. note::

        The final non-linear mapping :math:`f_{\mathbf{\Theta}}`, usually an :obj:`MLP`, is not included in this layer.
    """

    def __init__(self,
                 pooling_method='scatter_sum',
                 epsilon_learnable=False,
                 activation="relu",
                 activity_regularizer=None,
                 **kwargs):
        """Initialize layer.

        Args:
            epsilon_learnable (bool): If epsilon is learnable or just constant zero. Default is False.
            pooling_method (str): Pooling method for summing edges. Default is 'segment_sum'.
            activation: Activation function, such as `tf.nn.relu`, or string name of
                built-in activation function, such as "relu".
            activity_regularizer: Regularizer function applied to
                the output of the layer (its "activation"). Default is None.
        """
        super(GINE, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.epsilon_learnable = epsilon_learnable

        # Layers
        self.layer_gather = GatherNodesOutgoing()
        self.layer_pool = AggregateLocalEdges(pooling_method=self.pooling_method)
        self.layer_add = Add()
        self.layer_act = Activation(activation=activation,
                                    activity_regularizer=activity_regularizer)

        # Epsilon with trainable as optional and default zeros initialized.
        self.eps_k = self.add_weight(shape=tuple(), name="epsilon_k", trainable=self.epsilon_learnable,
                                     initializer="zeros", dtype=self.dtype)

    def build(self, input_shape):
        """Build layer."""
        super(GINE, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [nodes, edge_index, edges]

                - nodes (Tensor): Node embeddings of shape `([N], F)`
                - edge_index (Tensor): Edge indices referring to nodes of shape `(2, [M])`
                - edges (Tensor): Edge embeddings for index tensor of shape `([M], F)`

        Returns:
            Tensor: Node embeddings of shape `([N], F)`
        """
        node, edge_index, edges = inputs
        ed = self.layer_gather([node, edge_index], **kwargs)
        ed = self.layer_add([ed, edges])
        ed = self.layer_act(ed)
        nu = self.layer_pool([node, ed, edge_index], **kwargs)  # Summing for each node connection
        no = (ops.convert_to_tensor(1, dtype=self.eps_k.dtype) + self.eps_k)*node
        out = self.layer_add([no, nu], **kwargs)
        return out

    def get_config(self):
        """Update config."""
        config = super(GINE, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "epsilon_learnable": self.epsilon_learnable})
        conf_act = self.layer_act.get_config()
        for x in ["activation", "activity_regularizer"]:
            if x in conf_act:
                config.update({x: conf_act[x]})
        return config

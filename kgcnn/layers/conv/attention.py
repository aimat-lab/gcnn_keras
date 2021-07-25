import tensorflow as tf
# import tensorflow.keras as ks
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing, GatherState
from kgcnn.layers.keras import Dense, Activation, Concatenate
from kgcnn.layers.pool.pooling import PoolingNodes
from kgcnn.layers.pool.pooling import PoolingNodesAttention, PoolingLocalEdgesAttention


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='AttentionHeadGAT')
class AttentionHeadGAT(GraphBaseLayer):
    r"""Computes the attention head according to GAT from https://arxiv.org/abs/1710.10903.
    The attention coefficients are computed by :math:`a_{ij} = \sigma(a^T W n_i || W n_j)`,
    optionally by :math:`a_{ij} = \sigma( W n_i || W n_j || e_{ij})` with edges :math:`e_{ij}`.
    The attention is obtained by :math:`\alpha_{ij} = \text{softmax}_j (a_{ij})`.
    And the messages are pooled by :math:`m_i = \sum_j \alpha_{ij} W n_j`.
    If the graph has no self-loops, they must be added beforehand or use external skip connections.
    And optionally passed through an activation :math:`h_i = \sigma(\sum_j \alpha_{ij} W n_j)`.

    An edge is defined by index tuple (i,j) with i<-j connection.
    If graphs indices were in 'batch' mode, the layer's 'node_indexing' must be set to 'batch'.

    Args:
        units (int): Units for the linear trafo of node features before attention.
        use_edge_features (bool): Append edge features to attention computation. Default is False.
        use_final_activation (bool): Whether to apply the final activation for the output.
        has_self_loops (bool): If the graph has self-loops. Not used here. Default is True.
        activation (str): Activation. Default is {"class_name": "kgcnn>leaky_relu", "config": {"alpha": 0.2}},
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
                 use_edge_features=False,
                 use_final_activation=True,
                 has_self_loops=True,
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
        super(AttentionHeadGAT, self).__init__(**kwargs)
        # graph args
        self.use_edge_features = use_edge_features
        self.use_final_activation = use_final_activation
        self.has_self_loops = has_self_loops

        # dense args
        self.units = int(units)
        self.use_bias = use_bias

        kernel_args = {"kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}

        self.lay_linear_trafo = Dense(units, activation="linear", use_bias=use_bias, **kernel_args, **self._kgcnn_info)
        self.lay_alpha = Dense(1, activation=activation, use_bias=False, **kernel_args, **self._kgcnn_info)
        self.lay_gather_in = GatherNodesIngoing(**self._kgcnn_info)
        self.lay_gather_out = GatherNodesOutgoing(**self._kgcnn_info)
        self.lay_concat = Concatenate(axis=-1, **self._kgcnn_info)
        self.lay_pool_attention = PoolingLocalEdgesAttention(**self._kgcnn_info)
        self.lay_final_activ = Activation(activation=activation, **self._kgcnn_info)

    def build(self, input_shape):
        """Build layer."""
        super(AttentionHeadGAT, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, edges, edge_indices]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - edge_indices (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Embedding tensor of pooled edge attentions for each node.
        """
        node, edge, edge_index = inputs

        w_n = self.lay_linear_trafo(node)
        wn_in = self.lay_gather_in([w_n, edge_index])
        wn_out = self.lay_gather_out([w_n, edge_index])
        if self.use_edge_features:
            e_ij = self.lay_concat([wn_in, wn_out, edge])
        else:
            e_ij = self.lay_concat([wn_in, wn_out])
        a_ij = self.lay_alpha(e_ij)  # Should be dimension (batch*None,1)
        h_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index])

        if self.use_final_activation:
            h_i = self.lay_final_activ(h_i)
        return h_i

    def get_config(self):
        """Update layer config."""
        config = super(AttentionHeadGAT, self).get_config()
        config.update({"use_edge_features": self.use_edge_features, "use_bias": self.use_bias,
                       "units": self.units, "has_self_loops": self.has_self_loops,
                       "use_final_activation": self.use_final_activation})
        conf_sub = self.lay_alpha.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            config.update({x: conf_sub[x]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='AttentionHeadGATV2')
class AttentionHeadGATV2(GraphBaseLayer):
    r"""Computes the modified attention head according to https://arxiv.org/pdf/2105.14491.pdf.
    The attention coefficients are computed by :math:`a_{ij} = a^T \sigma( W [n_i || n_j] )`,
    optionally by :math:`a_{ij} = a^T \sigma( W [n_i || n_j || e_{ij}] )` with edges :math:`e_{ij}`.
    The attention is obtained by :math:`\alpha_{ij} = \text{softmax}_j (a_{ij})`.
    And the messages are pooled by :math:`m_i =  \sum_j \alpha_{ij} e_{ij}`.
    If the graph has no self-loops, they must be added beforehand or use external skip connections.
    And optionally passed through an activation :math:`h_i = \sigma(\sum_j \alpha_{ij} e_{ij})`.

    An edge is defined by index tuple (i,j) with i<-j connection.
    If graphs indices were in 'batch' mode, the layer's 'node_indexing' must be set to 'batch'.

    Args:
        units (int): Units for the linear trafo of node features before attention.
        use_edge_features (bool): Append edge features to attention computation. Default is False.
        use_final_activation (bool): Whether to apply the final activation for the output.
        has_self_loops (bool): If the graph has self-loops. Not used here. Default is True.
        activation (str): Activation. Default is {"class_name": "kgcnn>leaky_relu", "config": {"alpha": 0.2}},
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
                 use_edge_features=False,
                 use_final_activation=True,
                 has_self_loops=True,
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
        super(AttentionHeadGATV2, self).__init__(**kwargs)
        # graph args
        self.use_edge_features = use_edge_features
        self.use_final_activation = use_final_activation
        self.has_self_loops = has_self_loops

        # dense args
        self.units = int(units)
        self.use_bias = use_bias

        kernel_args = {"kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}

        self.lay_linear_trafo = Dense(units, activation="linear", use_bias=use_bias, **kernel_args, **self._kgcnn_info)
        self.lay_alpha_activation = Dense(units, activation=activation, use_bias=use_bias, **kernel_args,
                                          **self._kgcnn_info)
        self.lay_alpha = Dense(1, activation="linear", use_bias=False, **kernel_args, **self._kgcnn_info)
        self.lay_gather_in = GatherNodesIngoing(**self._kgcnn_info)
        self.lay_gather_out = GatherNodesOutgoing(**self._kgcnn_info)
        self.lay_concat = Concatenate(axis=-1, **self._kgcnn_info)
        self.lay_pool_attention = PoolingLocalEdgesAttention(**self._kgcnn_info)
        self.lay_final_activ = Activation(activation=activation, **self._kgcnn_info)

    def build(self, input_shape):
        """Build layer."""
        super(AttentionHeadGATV2, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, edges, edge_indices]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - edge_indices (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Embedding tensor of pooled edge attentions for each node.
        """
        node, edge, edge_index = inputs

        w_n = self.lay_linear_trafo(node)
        n_in = self.lay_gather_in([node, edge_index])
        n_out = self.lay_gather_out([node, edge_index])
        wn_out = self.lay_gather_out([w_n, edge_index])
        if self.use_edge_features:
            e_ij = self.lay_concat([n_in, n_out, edge])
        else:
            e_ij = self.lay_concat([n_in, n_out])
        a_ij = self.lay_alpha_activation(e_ij)
        a_ij = self.lay_alpha(a_ij)
        h_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index])

        if self.use_final_activation:
            h_i = self.lay_final_activ(h_i)
        return h_i

    def get_config(self):
        """Update layer config."""
        config = super(AttentionHeadGATV2, self).get_config()
        config.update({"use_edge_features": self.use_edge_features, "use_bias": self.use_bias,
                       "units": self.units, "has_self_loops": self.has_self_loops,
                       "use_final_activation": self.use_final_activation})
        conf_sub = self.lay_alpha_activation.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            config.update({x: conf_sub[x]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='AttentiveHeadFP')
class AttentiveHeadFP(GraphBaseLayer):
    r"""Computes the attention head for Attentive FP model.
    The attention coefficients are computed by :math:`a_{ij} = \sigma_1( W_1 [h_i || h_j] )`.
    The initial representation :math:`h_i` and :math:`h_j` must be calculated beforehand.
    The attention is obtained by :math:`\alpha_{ij} = \text{softmax}_j (a_{ij})`.
    And finally pooled for context :math:`C_i = \sigma_2(\sum_j \alpha_{ij} W_2 h_j)`.

    If graphs indices were in 'batch' mode, the layer's 'node_indexing' must be set to 'batch'.

    Args:
        units (int): Units for the linear trafo of node features before attention.
        use_edge_features (bool): Append edge features to attention computation. Default is False.
        activation (str): Activation. Default is {"class_name": "kgcnn>leaky_relu", "config": {"alpha": 0.2}}.
        activation_context (str): Activation function for context. Default is "elu".
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
                 use_edge_features=False,
                 activation='kgcnn>leaky_relu',
                 activation_context="elu",
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
        super(AttentiveHeadFP, self).__init__(**kwargs)
        # graph args
        self.use_edge_features = use_edge_features

        # dense args
        self.units = int(units)
        self.use_bias = use_bias

        kernel_args = {"kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}

        self.lay_linear_trafo = Dense(units, activation="linear", use_bias=use_bias, **kernel_args, **self._kgcnn_info)
        self.lay_alpha_activation = Dense(units, activation=activation, use_bias=use_bias, **kernel_args,
                                          **self._kgcnn_info)
        self.lay_alpha = Dense(1, activation="linear", use_bias=False, **kernel_args, **self._kgcnn_info)
        self.lay_gather_in = GatherNodesIngoing(**self._kgcnn_info)
        self.lay_gather_out = GatherNodesOutgoing(**self._kgcnn_info)
        self.lay_concat = Concatenate(axis=-1, **self._kgcnn_info)
        self.lay_pool_attention = PoolingLocalEdgesAttention(**self._kgcnn_info)
        self.lay_final_activ = Activation(activation=activation_context, **self._kgcnn_info)
        if use_edge_features:
            self.lay_fc1 = Dense(units, activation=activation, use_bias=use_bias, **kernel_args, **self._kgcnn_info)
            self.lay_fc2 = Dense(units, activation=activation, use_bias=use_bias, **kernel_args, **self._kgcnn_info)
            self.lay_concat_edge = Concatenate(axis=-1, **self._kgcnn_info)

    def build(self, input_shape):
        """Build layer."""
        super(AttentiveHeadFP, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, edges, edge_indices]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - edge_indices (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Hidden tensor of pooled edge attentions for each node.
        """
        node, edge, edge_index = inputs

        if self.use_edge_features:
            n_in = self.lay_gather_in([node, edge_index])
            n_out = self.lay_gather_out([node, edge_index])
            n_in = self.lay_fc1(n_in)
            n_out = self.lay_concat_edge([n_out, edge])
            n_out = self.lay_fc2(n_out)
        else:
            n_in = self.lay_gather_in([node, edge_index])
            n_out = self.lay_gather_out([node, edge_index])

        wn_out = self.lay_linear_trafo(n_out)
        e_ij = self.lay_concat([n_in, n_out])
        e_ij = self.lay_alpha_activation(e_ij)  # Maybe uses GAT original definition.
        # a_ij = e_ij
        a_ij = self.lay_alpha(e_ij)  # Should be dimension (batch,None,1) not fully clear in original paper SI.
        n_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index])
        out = self.lay_final_activ(n_i)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(AttentiveHeadFP, self).get_config()
        config.update({"use_edge_features": self.use_edge_features, "use_bias": self.use_bias,
                       "units": self.units})
        conf_sub = self.lay_alpha_activation.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            config.update({x: conf_sub[x]})
        conf_context = self.lay_final_activ.get_config()
        config.update({"activation_context": conf_context["activation"]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='PoolingNodesAttentive')
class PoolingNodesAttentive(GraphBaseLayer):
    r"""Computes the attentive pooling for node embeddings.

    Args:
        units (int): Units for the linear trafo of node features before attention.
        pooling_method(str): Initial pooling before iteration. Default is "sum".
        depth (int): Number of iterations for graph embedding. Default is 3.
        activation (str): Activation. Default is {"class_name": "kgcnn>leaky_relu", "config": {"alpha": 0.2}}.
        activation_context (str): Activation function for context. Default is "elu".
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
                 depth=3,
                 pooling_method="sum",
                 activation='kgcnn>leaky_relu',
                 activation_context="elu",
                 use_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 recurrent_activation='sigmoid',
                 recurrent_initializer='orthogonal',
                 recurrent_regularizer=None,
                 recurrent_constraint=None,
                 dropout=0.0,
                 recurrent_dropout=0.0,
                 reset_after=True,
                 **kwargs):
        """Initialize layer."""
        super(PoolingNodesAttentive, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.depth = depth
        # dense args
        self.units = int(units)

        kernel_args = {"use_bias": use_bias, "kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}
        gru_args = {"recurrent_activation": recurrent_activation,
                    "use_bias": use_bias, "kernel_initializer": kernel_initializer,
                    "recurrent_initializer": recurrent_initializer, "bias_initializer": bias_initializer,
                    "kernel_regularizer": kernel_regularizer, "recurrent_regularizer": recurrent_regularizer,
                    "bias_regularizer": bias_regularizer, "kernel_constraint": kernel_constraint,
                    "recurrent_constraint": recurrent_constraint, "bias_constraint": bias_constraint,
                    "dropout": dropout, "recurrent_dropout": recurrent_dropout, "reset_after": reset_after}

        self.lay_linear_trafo = Dense(units, activation="linear", **kernel_args, **self._kgcnn_info)
        self.lay_alpha = Dense(1, activation=activation, **kernel_args, **self._kgcnn_info)
        self.lay_gather_s = GatherState(**self._kgcnn_info)
        self.lay_concat = Concatenate(axis=-1, **self._kgcnn_info)
        self.lay_pool_start = PoolingNodes(pooling_method=self.pooling_method, **self._kgcnn_info)
        self.lay_pool_attention = PoolingNodesAttention(**self._kgcnn_info)
        self.lay_final_activ = Activation(activation=activation_context, **self._kgcnn_info)
        self.lay_gru = tf.keras.layers.GRUCell(units=units, activation="tanh", **gru_args)

    def build(self, input_shape):
        """Build layer."""
        super(PoolingNodesAttentive, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: nodes

                - nodes (tf.RaggedTensor): Node features of shape (batch, [N], F)

        Returns:
            tf.Tensor: Hidden tensor of pooled node attentions of shape (batch, F).
        """
        node = inputs

        h = self.lay_pool_start(node)
        wn = self.lay_linear_trafo(node)
        for _ in range(self.depth):
            hv = self.lay_gather_s([h, node])
            ev = self.lay_concat([hv, node])
            av = self.lay_alpha(ev)
            cont = self.lay_pool_attention([wn, av])
            cont = self.lay_final_activ(cont)
            h, _ = self.lay_gru(cont, h, **kwargs)

        out = h
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingNodesAttentive, self).get_config()
        config.update({"units": self.units, "depth": self.depth, "pooling_method": self.pooling_method})
        conf_sub = self.lay_alpha.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias"]:
            config.update({x: conf_sub[x]})
        conf_context = self.lay_final_activ.get_config()
        config.update({"activation_context": conf_context["activation"]})
        conf_gru = self.lay_gru.get_config()
        for x in ["recurrent_activation", "recurrent_initializer", "recurrent_regularizer", "recurrent_constraint",
                  "dropout", "recurrent_dropout", "reset_after"]:
            config.update({x: conf_gru[x]})
        return config

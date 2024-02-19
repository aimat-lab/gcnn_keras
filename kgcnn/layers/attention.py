# import keras_core as ks
from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing
from keras.layers import Dense, Concatenate, Activation, Average, Layer
from kgcnn.layers.aggr import AggregateLocalEdgesAttention
from keras import ops
import kgcnn.ops.activ


class AttentionHeadGAT(Layer):  # noqa
    r"""Computes the attention head according to `GAT <https://arxiv.org/abs/1710.10903>`__ .

    The attention coefficients are computed by :math:`a_{ij} = \sigma(a^T W n_i || W n_j)`,
    optionally by :math:`a_{ij} = \sigma( W n_i || W n_j || e_{ij})` with edges :math:`e_{ij}`.
    The attention is obtained by :math:`\alpha_{ij} = \text{softmax}_j (a_{ij})`.
    And the messages are pooled by :math:`m_i = \sum_j \alpha_{ij} W n_j`.
    If the graph has no self-loops, they must be added beforehand or use external skip connections.
    And optionally passed through an activation :math:`h_i = \sigma(\sum_j \alpha_{ij} W n_j)`.

    An edge is defined by index tuple :math:`(i, j)` with the direction of the connection from :math:`j` to :math:`i`.
    """

    def __init__(self,
                 units,
                 use_edge_features=False,
                 use_final_activation=True,
                 has_self_loops=True,
                 activation="kgcnn>leaky_relu2",
                 use_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 normalize_softmax: bool = False,
                 **kwargs):
        """Initialize layer.

        Args:
            units (int): Units for the linear trafo of node features before attention.
            use_edge_features (bool): Append edge features to attention computation. Default is False.
            use_final_activation (bool): Whether to apply the final activation for the output.
            has_self_loops (bool): If the graph has self-loops. Not used here. Default is True.
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
        super(AttentionHeadGAT, self).__init__(**kwargs)
        # Changes in keras serialization behaviour for activations in 3.0.2.
        # Keep string at least for default. Also renames to prevent clashes with keras leaky_relu.
        if activation in ["kgcnn>leaky_relu", "kgcnn>leaky_relu2"]:
            activation = {"class_name": "function", "config": "kgcnn>leaky_relu2"}
        self.use_edge_features = use_edge_features
        self.use_final_activation = use_final_activation
        self.has_self_loops = has_self_loops
        self.normalize_softmax = normalize_softmax
        self.units = int(units)
        self.use_bias = use_bias
        kernel_args = {"kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}

        self.lay_linear_trafo = Dense(units, activation="linear", use_bias=use_bias, **kernel_args)
        self.lay_alpha = Dense(1, activation=activation, use_bias=False, **kernel_args)
        self.lay_gather_in = GatherNodesIngoing()
        self.lay_gather_out = GatherNodesOutgoing()
        self.lay_concat = Concatenate(axis=-1)
        self.lay_pool_attention = AggregateLocalEdgesAttention(normalize_softmax=normalize_softmax)
        if self.use_final_activation:
            self.lay_final_activ = Activation(activation=activation)

    def build(self, input_shape):
        """Build layer."""
        super(AttentionHeadGAT, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, edges, edge_indices]

                - nodes (Tensor): Node embeddings of shape ([N], F)
                - edges (Tensor): Edge or message embeddings of shape ([M], F)
                - edge_indices (Tensor): Edge indices referring to nodes of shape (2, [M])

        Returns:
            Tensor: Embedding tensor of pooled edge attentions for each node.
        """
        node, edge, edge_index = inputs

        w_n = self.lay_linear_trafo(node, **kwargs)
        wn_in = self.lay_gather_in([w_n, edge_index], **kwargs)
        wn_out = self.lay_gather_out([w_n, edge_index], **kwargs)
        if self.use_edge_features:
            e_ij = self.lay_concat([wn_in, wn_out, edge], **kwargs)
        else:
            e_ij = self.lay_concat([wn_in, wn_out], **kwargs)
        a_ij = self.lay_alpha(e_ij, **kwargs)  # Should be dimension (batch, None,1)
        h_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index], **kwargs)

        if self.use_final_activation:
            h_i = self.lay_final_activ(h_i, **kwargs)
        return h_i

    def get_config(self):
        """Update layer config."""
        config = super(AttentionHeadGAT, self).get_config()
        config.update({"use_edge_features": self.use_edge_features, "use_bias": self.use_bias,
                       "units": self.units, "has_self_loops": self.has_self_loops,
                       "normalize_softmax": self.normalize_softmax,
                       "use_final_activation": self.use_final_activation})
        conf_sub = self.lay_alpha.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            if x in conf_sub:
                config.update({x: conf_sub[x]})
        return config


class AttentionHeadGATV2(Layer):  # noqa
    r"""Computes the modified attention head according to `GATv2 <https://arxiv.org/pdf/2105.14491.pdf>`__ .

    The attention coefficients are computed by :math:`a_{ij} = a^T \sigma( W [n_i || n_j] )`,
    optionally by :math:`a_{ij} = a^T \sigma( W [n_i || n_j || e_{ij}] )` with edges :math:`e_{ij}`.
    The attention is obtained by :math:`\alpha_{ij} = \text{softmax}_j (a_{ij})`.
    And the messages are pooled by :math:`m_i =  \sum_j \alpha_{ij} e_{ij}`.
    If the graph has no self-loops, they must be added beforehand or use external skip connections.
    And optionally passed through an activation :math:`h_i = \sigma(\sum_j \alpha_{ij} e_{ij})`.

    An edge is defined by index tuple :math:`(i, j)` with the direction of the connection from :math:`j` to :math:`i`.
    """

    def __init__(self,
                 units,
                 use_edge_features=False,
                 use_final_activation=True,
                 has_self_loops=True,
                 activation="kgcnn>leaky_relu2",
                 use_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 normalize_softmax: bool = False,
                 **kwargs):
        """Initialize layer.

        Args:
            units (int): Units for the linear trafo of node features before attention.
            use_edge_features (bool): Append edge features to attention computation. Default is False.
            use_final_activation (bool): Whether to apply the final activation for the output.
            has_self_loops (bool): If the graph has self-loops. Not used here. Default is True.
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
        super(AttentionHeadGATV2, self).__init__(**kwargs)
        # Changes in keras serialization behaviour for activations in 3.0.2.
        # Keep string at least for default. Also renames to prevent clashes with keras leaky_relu.
        if activation in ["kgcnn>leaky_relu", "kgcnn>leaky_relu2"]:
            activation = {"class_name": "function", "config": "kgcnn>leaky_relu2"}
        self.use_edge_features = use_edge_features
        self.use_final_activation = use_final_activation
        self.has_self_loops = has_self_loops
        self.units = int(units)
        self.normalize_softmax = normalize_softmax
        self.use_bias = use_bias
        kernel_args = {"kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}

        self.lay_linear_trafo = Dense(units, activation="linear", use_bias=use_bias, **kernel_args)
        self.lay_alpha_activation = Dense(units, activation=activation, use_bias=use_bias, **kernel_args)
        self.lay_alpha = Dense(1, activation="linear", use_bias=False, **kernel_args)
        self.lay_gather_in = GatherNodesIngoing()
        self.lay_gather_out = GatherNodesOutgoing()
        self.lay_concat = Concatenate(axis=-1)
        self.lay_pool_attention = AggregateLocalEdgesAttention(normalize_softmax=normalize_softmax)
        if self.use_final_activation:
            self.lay_final_activ = Activation(activation=activation)

    def build(self, input_shape):
        """Build layer."""
        super(AttentionHeadGATV2, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, edges, edge_indices]

                - nodes (Tensor): Node embeddings of shape ([N], F)
                - edges (Tensor): Edge or message embeddings of shape ([M], F)
                - edge_indices (Tensor): Edge indices referring to nodes of shape (2, [M])

        Returns:
            Tensor: Embedding tensor of pooled edge attentions for each node.
        """
        node, edge, edge_index = inputs

        w_n = self.lay_linear_trafo(node, **kwargs)
        n_in = self.lay_gather_in([node, edge_index], **kwargs)
        n_out = self.lay_gather_out([node, edge_index], **kwargs)
        wn_out = self.lay_gather_out([w_n, edge_index], **kwargs)
        if self.use_edge_features:
            e_ij = self.lay_concat([n_in, n_out, edge], **kwargs)
        else:
            e_ij = self.lay_concat([n_in, n_out], **kwargs)
        a_ij = self.lay_alpha_activation(e_ij, **kwargs)
        a_ij = self.lay_alpha(a_ij, **kwargs)
        h_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index], **kwargs)

        if self.use_final_activation:
            h_i = self.lay_final_activ(h_i, **kwargs)
        return h_i

    def get_config(self):
        """Update layer config."""
        config = super(AttentionHeadGATV2, self).get_config()
        config.update({"use_edge_features": self.use_edge_features, "use_bias": self.use_bias,
                       "units": self.units, "has_self_loops": self.has_self_loops,
                       "normalize_softmax": self.normalize_softmax,
                       "use_final_activation": self.use_final_activation})
        conf_sub = self.lay_alpha_activation.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            if x in conf_sub:
                config.update({x: conf_sub[x]})
        return config


class MultiHeadGATV2Layer(Layer):  # noqa
    r"""Single layer for multiple Attention heads from :obj:`AttentionHeadGATV2` .

    Uses concatenation or averaging of heads for final output.
    """

    def __init__(self,
                 units: int,
                 num_heads: int,
                 activation: str = "kgcnn>leaky_relu2",
                 use_bias: bool = True,
                 concat_heads: bool = True,
                 use_edge_features=False,
                 use_final_activation=True,
                 has_self_loops=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 normalize_softmax: bool = False,
                 **kwargs):
        r"""Initialize layer.

        Args:
            units (int): Units for the linear trafo of node features before attention.
            num_heads: Number of attention heads.
            concat_heads: Whether to concatenate heads or average.
            use_edge_features (bool): Append edge features to attention computation. Default is False.
            use_final_activation (bool): Whether to apply the final activation for the output.
            has_self_loops (bool): If the graph has self-loops. Not used here. Default is True.
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
        super(MultiHeadGATV2Layer, self).__init__(**kwargs)
        # Changes in keras serialization behaviour for activations in 3.0.2.
        # Keep string at least for default. Also renames to prevent clashes with keras leaky_relu.
        if activation in ["kgcnn>leaky_relu", "kgcnn>leaky_relu2"]:
            activation = {"class_name": "function", "config": "kgcnn>leaky_relu2"}
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.use_edge_features = use_edge_features
        self.use_final_activation = use_final_activation
        self.has_self_loops = has_self_loops
        self.units = int(units)
        self.normalize_softmax = normalize_softmax
        self.use_bias = use_bias
        kernel_args = {"kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}

        self.head_layers = []
        for _ in range(num_heads):
            lay_linear = Dense(units, activation=activation, use_bias=use_bias, **kernel_args)
            lay_alpha_activation = Dense(units, activation=activation, use_bias=use_bias, **kernel_args)
            lay_alpha = Dense(1, activation='linear', use_bias=False, **kernel_args)

            self.head_layers.append((lay_linear, lay_alpha_activation, lay_alpha))

        self.lay_concat_alphas = Concatenate(axis=-2)

        # self.lay_linear_trafo = Dense(units, activation="linear", use_bias=use_bias, **kernel_args)
        # self.lay_alpha_activation = Dense(units, activation=activation, use_bias=use_bias, **kernel_args)
        # self.lay_alpha = Dense(1, activation="linear", use_bias=False, **kernel_args)
        self.lay_gather_in = GatherNodesIngoing()
        self.lay_gather_out = GatherNodesOutgoing()
        self.lay_concat = Concatenate(axis=-1)
        self.lay_pool_attention = AggregateLocalEdgesAttention(normalize_softmax=normalize_softmax)
        if self.use_final_activation:
            self.lay_final_activ = Activation(activation=activation)

        if self.concat_heads:
            self.lay_combine_heads = Concatenate(axis=-1)
        else:
            self.lay_combine_heads = Average()

    def build(self, input_shape):
        """Build layer."""
        super(MultiHeadGATV2Layer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        node, edge, edge_index = inputs

        # "a_ij" is a single-channel edge attention logits tensor. "a_ijs" is consequently the list which
        # stores these tensors for each attention head.
        # "h_i" is a single-channel node embedding tensor. "h_is" is consequently the list which stores
        # these tensors for each attention head.
        a_ijs = []
        h_is = []
        for k, (lay_linear, lay_alpha_activation, lay_alpha) in enumerate(self.head_layers):

            # Copied from the original class
            w_n = lay_linear(node, **kwargs)
            n_in = self.lay_gather_in([node, edge_index], **kwargs)
            n_out = self.lay_gather_out([node, edge_index], **kwargs)
            wn_out = self.lay_gather_out([w_n, edge_index], **kwargs)
            if self.use_edge_features:
                e_ij = self.lay_concat([n_in, n_out, edge], **kwargs)
            else:
                e_ij = self.lay_concat([n_in, n_out], **kwargs)

            # a_ij: ([batch], [M], 1)
            a_ij = lay_alpha_activation(e_ij, **kwargs)
            a_ij = lay_alpha(a_ij, **kwargs)

            # h_i: ([batch], [N], F)
            h_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index], **kwargs)

            if self.use_final_activation:
                h_i = self.lay_final_activ(h_i, **kwargs)

            # a_ij after expand: ([batch], [M], 1, 1)
            a_ij = ops.expand_dims(a_ij, axis=-2)
            a_ijs.append(a_ij)

            # h_i = tf.expand_dims(h_i, axis=-2)
            h_is.append(h_i)

        a_ijs = self.lay_concat_alphas(a_ijs)

        h_is = self.lay_combine_heads(h_is)

        # An important modification we need here is that this layer also returns the attention coefficients
        # because in MEGAN we need those to calculate the edge attention values with!
        # h_is: ([batch], [N], K * Vu) or ([batch], [N], Vu)
        # a_ijs: ([batch], [M], K, 1)
        return h_is, a_ijs

    def get_config(self):
        """Update layer config."""
        config = super(MultiHeadGATV2Layer, self).get_config()
        config.update({"use_edge_features": self.use_edge_features, "use_bias": self.use_bias,
                       "units": self.units, "has_self_loops": self.has_self_loops,
                       "normalize_softmax": self.normalize_softmax,
                       "use_final_activation": self.use_final_activation})
        if self.num_heads > 0:
            conf_sub = self.head_layers[0][0].get_config()
            for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                      "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
                if x in conf_sub:
                    config.update({x: conf_sub[x]})
        config.update({
            'num_heads': self.num_heads,
            'concat_heads': self.concat_heads
        })
        return config


class AttentiveHeadFP(Layer):
    r"""Computes the attention head for `Attentive FP <https://doi.org/10.1021/acs.jmedchem.9b00959>`__ model.
    The attention coefficients are computed by :math:`a_{ij} = \sigma_1( W_1 [h_i || h_j] )`.
    The initial representation :math:`h_i` and :math:`h_j` must be calculated beforehand.
    The attention is obtained by :math:`\alpha_{ij} = \text{softmax}_j (a_{ij})`.
    And finally pooled for context :math:`C_i = \sigma_2(\sum_j \alpha_{ij} W_2 h_j)`.

    An edge is defined by index tuple :math:`(i, j)` with the direction of the connection from :math:`j` to :math:`i`.
    """

    def __init__(self,
                 units,
                 use_edge_features=False,
                 activation="kgcnn>leaky_relu2",
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
        """Initialize layer.

        Args:
            units (int): Units for the linear trafo of node features before attention.
            use_edge_features (bool): Append edge features to attention computation. Default is False.
            activation (str): Activation. Default is "kgcnn>leaky_relu2".
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
        super(AttentiveHeadFP, self).__init__(**kwargs)
        # Changes in keras serialization behaviour for activations in 3.0.2.
        # Keep string at least for default. Also renames to prevent clashes with keras leaky_relu.
        if activation in ["kgcnn>leaky_relu", "kgcnn>leaky_relu2"]:
            activation = {"class_name": "function", "config": "kgcnn>leaky_relu2"}
        self.use_edge_features = use_edge_features
        self.units = int(units)
        self.use_bias = use_bias
        kernel_args = {"kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}

        self.lay_linear_trafo = Dense(units, activation="linear", use_bias=use_bias, **kernel_args)
        self.lay_alpha_activation = Dense(units, activation=activation, use_bias=use_bias, **kernel_args)
        self.lay_alpha = Dense(1, activation="linear", use_bias=False, **kernel_args)
        self.lay_gather_in = GatherNodesIngoing()
        self.lay_gather_out = GatherNodesOutgoing()
        self.lay_concat = Concatenate(axis=-1)
        self.lay_pool_attention = AggregateLocalEdgesAttention()
        self.lay_final_activ = Activation(activation=activation_context)
        if use_edge_features:
            self.lay_fc1 = Dense(units, activation=activation, use_bias=use_bias, **kernel_args)
            self.lay_fc2 = Dense(units, activation=activation, use_bias=use_bias, **kernel_args)
            self.lay_concat_edge = Concatenate(axis=-1)

    def build(self, input_shape):
        """Build layer."""
        super(AttentiveHeadFP, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [node, edges, edge_indices]

                - nodes (Tensor): Node embeddings of shape ([N], F)
                - edges (Tensor): Edge or message embeddings of shape ([M], F)
                - edge_indices (Tensor): Edge indices referring to nodes of shape ([M], 2)

        Returns:
            Tensor: Hidden tensor of pooled edge attentions for each node.
        """
        node, edge, edge_index = inputs

        if self.use_edge_features:
            n_in = self.lay_gather_in([node, edge_index], **kwargs)
            n_out = self.lay_gather_out([node, edge_index], **kwargs)
            n_in = self.lay_fc1(n_in, **kwargs)
            n_out = self.lay_concat_edge([n_out, edge], **kwargs)
            n_out = self.lay_fc2(n_out, **kwargs)
        else:
            n_in = self.lay_gather_in([node, edge_index], **kwargs)
            n_out = self.lay_gather_out([node, edge_index], **kwargs)

        wn_out = self.lay_linear_trafo(n_out, **kwargs)
        e_ij = self.lay_concat([n_in, n_out], **kwargs)
        e_ij = self.lay_alpha_activation(e_ij, **kwargs)  # Maybe uses GAT original definition.
        # a_ij = e_ij
        a_ij = self.lay_alpha(e_ij, **kwargs)  # Should be dimension (None, 1) not fully clear in original paper.
        n_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index], **kwargs)
        out = self.lay_final_activ(n_i, **kwargs)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(AttentiveHeadFP, self).get_config()
        config.update({"use_edge_features": self.use_edge_features, "use_bias": self.use_bias,
                       "units": self.units})
        conf_sub = self.lay_alpha_activation.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            if x in conf_sub.keys():
                config.update({x: conf_sub[x]})
        conf_context = self.lay_final_activ.get_config()
        config.update({"activation_context": conf_context["activation"]})
        return config

from keras.layers import Dense, Concatenate, Activation, Layer
from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.layers.aggr import AggregateLocalEdgesAttention


class AttentiveHeadFP_(Layer):
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
                 activation='kgcnn>leaky_relu2',
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
        super(AttentiveHeadFP_, self).__init__(**kwargs)
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
        super(AttentiveHeadFP_, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, edges, edge_indices]

                - nodes (Tensor): Node embeddings of shape ([N], F)
                - edges (Tensor): Edge or message embeddings of shape ([M], F)
                - edge_indices (Tensor): Edge indices referring to nodes of shape (2, [M])

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
        a_ij = self.lay_alpha(e_ij, **kwargs)  # Should be dimension (batch,None,1) not fully clear in original paper.
        n_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index], **kwargs)
        out = self.lay_final_activ(n_i, **kwargs)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(AttentiveHeadFP_, self).get_config()
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
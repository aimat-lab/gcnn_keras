import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.layers.modules import DenseEmbedding, LazyConcatenate, ActivationEmbedding
from kgcnn.layers.pooling import PoolingLocalEdgesAttention


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='AttentionHeadGAT')
class AttentionHeadGAT(GraphBaseLayer):
    r"""Computes the attention head according to `GAT <https://arxiv.org/abs/1710.10903>`_ .
    The attention coefficients are computed by :math:`a_{ij} = \sigma(a^T W n_i || W n_j)`,
    optionally by :math:`a_{ij} = \sigma( W n_i || W n_j || e_{ij})` with edges :math:`e_{ij}`.
    The attention is obtained by :math:`\alpha_{ij} = \text{softmax}_j (a_{ij})`.
    And the messages are pooled by :math:`m_i = \sum_j \alpha_{ij} W n_j`.
    If the graph has no self-loops, they must be added beforehand or use external skip connections.
    And optionally passed through an activation :math:`h_i = \sigma(\sum_j \alpha_{ij} W n_j)`.

    An edge is defined by index tuple :math:`(i, j)` with the direction of the connection from :math:`j` to :math:`i`.

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
        self.use_edge_features = use_edge_features
        self.use_final_activation = use_final_activation
        self.has_self_loops = has_self_loops
        self.units = int(units)
        self.use_bias = use_bias
        kernel_args = {"kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}

        self.lay_linear_trafo = DenseEmbedding(units, activation="linear", use_bias=use_bias, **kernel_args)
        self.lay_alpha = DenseEmbedding(1, activation=activation, use_bias=False, **kernel_args)
        self.lay_gather_in = GatherNodesIngoing()
        self.lay_gather_out = GatherNodesOutgoing()
        self.lay_concat = LazyConcatenate(axis=-1)
        self.lay_pool_attention = PoolingLocalEdgesAttention()
        if self.use_final_activation:
            self.lay_final_activ = ActivationEmbedding(activation=activation)

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

        w_n = self.lay_linear_trafo(node, **kwargs)
        wn_in = self.lay_gather_in([w_n, edge_index], **kwargs)
        wn_out = self.lay_gather_out([w_n, edge_index], **kwargs)
        if self.use_edge_features:
            e_ij = self.lay_concat([wn_in, wn_out, edge], **kwargs)
        else:
            e_ij = self.lay_concat([wn_in, wn_out], **kwargs)
        a_ij = self.lay_alpha(e_ij, **kwargs)  # Should be dimension (batch*None,1)
        h_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index], **kwargs)

        if self.use_final_activation:
            h_i = self.lay_final_activ(h_i, **kwargs)
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
    r"""Computes the modified attention head according to `GATv2 <https://arxiv.org/pdf/2105.14491.pdf>`_ .
    The attention coefficients are computed by :math:`a_{ij} = a^T \sigma( W [n_i || n_j] )`,
    optionally by :math:`a_{ij} = a^T \sigma( W [n_i || n_j || e_{ij}] )` with edges :math:`e_{ij}`.
    The attention is obtained by :math:`\alpha_{ij} = \text{softmax}_j (a_{ij})`.
    And the messages are pooled by :math:`m_i =  \sum_j \alpha_{ij} e_{ij}`.
    If the graph has no self-loops, they must be added beforehand or use external skip connections.
    And optionally passed through an activation :math:`h_i = \sigma(\sum_j \alpha_{ij} e_{ij})`.

    An edge is defined by index tuple :math:`(i, j)` with the direction of the connection from :math:`j` to :math:`i`.

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
        self.use_edge_features = use_edge_features
        self.use_final_activation = use_final_activation
        self.has_self_loops = has_self_loops
        self.units = int(units)
        self.use_bias = use_bias
        kernel_args = {"kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}

        self.lay_linear_trafo = DenseEmbedding(units, activation="linear", use_bias=use_bias, **kernel_args)
        self.lay_alpha_activation = DenseEmbedding(units, activation=activation, use_bias=use_bias, **kernel_args)
        self.lay_alpha = DenseEmbedding(1, activation="linear", use_bias=False, **kernel_args)
        self.lay_gather_in = GatherNodesIngoing()
        self.lay_gather_out = GatherNodesOutgoing()
        self.lay_concat = LazyConcatenate(axis=-1)
        self.lay_pool_attention = PoolingLocalEdgesAttention()
        if self.use_final_activation:
            self.lay_final_activ = ActivationEmbedding(activation=activation)

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
                       "use_final_activation": self.use_final_activation})
        conf_sub = self.lay_alpha_activation.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation"]:
            config.update({x: conf_sub[x]})
        return config
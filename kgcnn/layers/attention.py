import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing, GatherState
from kgcnn.layers.keras import Dense, Activation, Concatenate
from kgcnn.layers.pooling import PoolingNodes
import kgcnn.ops.activ
from kgcnn.ops.partition import kgcnn_ops_change_edge_tensor_indexing_by_row_partition
from kgcnn.ops.scatter import kgcnn_ops_scatter_segment_tensor_nd
from kgcnn.ops.segment import segment_softmax, kgcnn_ops_segment_operation_by_name


# import tensorflow.keras as ks

@tf.keras.utils.register_keras_serializable(package='kgcnn',name='PoolingLocalEdgesAttention')
class PoolingLocalEdgesAttention(GraphBaseLayer):
    r"""Pooling all edges or edge-like features per node, corresponding to node assigned by edge indices.
    Uses attention for pooling. i.e.  $n_i =  \sum_j \alpha_{ij} e_ij $
    The attention is computed via: $\alpha_ij = softmax_j (a_ij)$ from the attention coefficients $a_ij$.
    The attention coefficients must be computed beforehand by edge features or by $\sigma( W n_i || W n_j)$ and
    are passed to this layer as input. Thereby this layer has no weights and only does pooling.
    In summary, $n_i =  \sum_j softmax_j(a_ij) e_ij $ is computed by the layer.

    An edge is defined by index tuple (i,j) with i<-j connection.
    If graphs indices were in 'batch' mode, the layer's 'node_indexing' must be set to 'batch'.
    Important: edge_index[:,0] are sorted for segment-operation for pooling with respect to edge_index[:,0].
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(PoolingLocalEdgesAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(PoolingLocalEdgesAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [node, edges, attention, edge_indices]

            - nodes (tf.ragged): Node features of shape (batch, [N], F)
            - edges (tf.ragged): Edge or message features of shape (batch, [M], F)
            - attention (tf.ragged): Attention coefficients of shape (batch, [M], 1)
            - edge_index (tf.ragged): Edge indices of shape (batch, [M], F)

        Returns:
            embeddings: Feature tensor of pooled edge attentions for each node of shape (batch, [N], F)
        """
        dyn_inputs = self._kgcnn_map_input_ragged(inputs, 4)

        # We cast to values here
        nod, node_part = dyn_inputs[0].values, dyn_inputs[0].row_lengths()
        edge = dyn_inputs[1].values
        attention = dyn_inputs[2].values
        edgeind, edge_part = dyn_inputs[3].values, dyn_inputs[3].row_lengths()

        shiftind = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edgeind, node_part, edge_part,
                                                                          partition_type_node="row_length",
                                                                          partition_type_edge="row_length",
                                                                          to_indexing='batch',
                                                                          from_indexing=self.node_indexing)

        nodind = shiftind[:, 0]  # Pick first index eg. ingoing
        dens = edge
        ats = attention
        if not self.is_sorted:
            # Sort edgeindices
            node_order = tf.argsort(nodind, axis=0, direction='ASCENDING', stable=True)
            nodind = tf.gather(nodind, node_order, axis=0)
            dens = tf.gather(dens, node_order, axis=0)
            ats = tf.gather(ats, node_order, axis=0)

        # Apply segmented softmax
        ats = segment_softmax(ats, nodind)
        get = dens * ats
        get = tf.math.segment_sum(get, nodind)

        if self.has_unconnected:
            # Need to fill tensor since the maximum node may not be also in pooled
            # Does not happen if all nodes are also connected
            get = kgcnn_ops_scatter_segment_tensor_nd(get, nodind, tf.shape(nod))

        out = self._kgcnn_map_output_ragged([get, node_part], "row_length", 0)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingLocalEdgesAttention, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn',name='AttentionHeadGAT')
class AttentionHeadGAT(GraphBaseLayer):
    r"""Computes the attention head according to GAT.
    The attention coefficients are computed by $a_{ij} = \sigma( W n_i || W n_j)$,
    optionally by $a_{ij} = \sigma( W n_i || W n_j || e_{ij})$.
    The attention is obtained by $\alpha_ij = softmax_j (a_{ij})$.
    And the messages are pooled by $n_i =  \sum_j \alpha_{ij} e_ij $.
    And finally passed through an activation $h_i = \sigma(\sum_j \alpha_{ij} e_ij)$.

    An edge is defined by index tuple (i,j) with i<-j connection.
    If graphs indices were in 'batch' mode, the layer's 'node_indexing' must be set to 'batch'.

    Args:
        units (int): Units for the linear trafo of node features before attention.
        use_edge_features (bool): Append edge features to attention computation. Default is False.
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

        # dense args
        self.units = int(units)

        kernel_args = {"use_bias": use_bias, "kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}

        self.lay_linear_trafo = Dense(units, activation="linear", **kernel_args, **self._kgcnn_info)
        self.lay_alpha = Dense(1, activation=activation, **kernel_args, **self._kgcnn_info)
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

            - nodes: Node features of shape (batch, [N], F)
            - edges: Edge or message features of shape (batch, [M], F)
            - edge_index: Edge indices of shape (batch, [M], 2)

        Returns:
            features: Feature tensor of pooled edge attentions for each node.
        """
        node, edge, edge_index = inputs

        n_in = self.lay_gather_in([node, edge_index])
        n_out = self.lay_gather_out([node, edge_index])
        wn_in = self.lay_linear_trafo(n_in)
        wn_out = self.lay_linear_trafo(n_out)
        if self.use_edge_features:
            e_ij = self.lay_concat([wn_in, wn_out, edge])
        else:
            e_ij = self.lay_concat([wn_in, wn_out])
        a_ij = self.lay_alpha(e_ij)  # Should be dimension (batch*None,1)
        n_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index])
        out = self.lay_final_activ(n_i)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(AttentionHeadGAT, self).get_config()
        config.update({"use_edge_features": self.use_edge_features,
                       "units": self.units})
        conf_sub = self.lay_alpha.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias"]:
            config.update({x: conf_sub[x]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn',name='AttentiveHeadFP')
class AttentiveHeadFP(GraphBaseLayer):
    r"""Computes the attention head for Attentive FP model.
    The attention coefficients are computed by $a_{ij} = \sigma_1( W_1 [h_i || h_j] )$.
    The initial representation $h_i$ and $h_j$ must be calculated beforehand.
    The attention is obtained by $\alpha_ij = softmax_j (a_{ij})$.
    And finally pooled through for context $C_i = \sigma_2(\sum_j \alpha_{ij} W_2 h_j)$.

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

        kernel_args = {"use_bias": use_bias, "kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}

        self.lay_linear_trafo = Dense(units, activation="linear", **kernel_args, **self._kgcnn_info)
        self.lay_alpha = Dense(1, activation=activation, **kernel_args, **self._kgcnn_info)
        self.lay_gather_in = GatherNodesIngoing(**self._kgcnn_info)
        self.lay_gather_out = GatherNodesOutgoing(**self._kgcnn_info)
        self.lay_concat = Concatenate(axis=-1, **self._kgcnn_info)
        self.lay_pool_attention = PoolingLocalEdgesAttention(**self._kgcnn_info)
        self.lay_final_activ = Activation(activation=activation_context, **self._kgcnn_info)
        if use_edge_features:
            self.lay_fc1 = Dense(units, activation=activation, **kernel_args, **self._kgcnn_info)
            self.lay_fc2 = Dense(units, activation=activation, **kernel_args, **self._kgcnn_info)
            self.lay_concat_edge = Concatenate(axis=-1, **self._kgcnn_info)

    def build(self, input_shape):
        """Build layer."""
        super(AttentiveHeadFP, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, edges, edge_indices]

            - nodes: Node features of shape (batch, [N], F)
            - edges: Edge or message features of shape (batch, [M], F)
            - edge_index: Edge indices of shape (batch, [M], 2)

        Returns:
            features: Hidden tensor of pooled edge attentions for each node.
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
        a_ij = self.lay_alpha(e_ij)  # Should be dimension (batch,None,1)
        n_i = self.lay_pool_attention([node, wn_out, a_ij, edge_index])
        out = self.lay_final_activ(n_i)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(AttentiveHeadFP, self).get_config()
        config.update({"use_edge_features": self.use_edge_features,
                       "units": self.units})
        conf_sub = self.lay_alpha.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias"]:
            config.update({x: conf_sub[x]})
        conf_context = self.lay_final_activ.get_config()
        config.update({"activation_context": conf_context["activation"]})
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn',name='PoolingNodesAttention')
class PoolingNodesAttention(GraphBaseLayer):
    r"""Pooling all nodes
    Uses attention for pooling. i.e.  $s =  \sum_j \alpha_{i} n_i $
    The attention is computed via: $\alpha_i = softmax_i (a_i)$ from the attention coefficients $a_i$.
    The attention coefficients must be computed beforehand by edge features or by $\sigma( W [s || n_i])$ and
    are passed to this layer as input. Thereby this layer has no weights and only does pooling.
    In summary, $s =  \sum_i softmax_j(a_i) n_i $ is computed by the layer.

    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(PoolingNodesAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(PoolingNodesAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, attention]

            - nodes (tf.ragged): Node features of shape (batch, [N], F)
            - attention (tf.ragged): Attention coefficients of shape (batch, [N], 1)

        Returns:
            embeddings: Feature tensor of pooled node of shape (batch, F)
        """
        dyn_inputs = self._kgcnn_map_input_ragged(inputs, 2)
        # We cast to values here
        nod, batchi, target_len = dyn_inputs[0].values, dyn_inputs[0].value_rowids(), dyn_inputs[0].row_lengths()
        ats = dyn_inputs[1].values

        ats = segment_softmax(ats, batchi)
        get = nod * ats
        out = tf.math.segment_sum(get, batchi)

        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingNodesAttention, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable(package='kgcnn',name='AttentiveNodePooling')
class AttentiveNodePooling(GraphBaseLayer):
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
        super(AttentiveNodePooling, self).__init__(**kwargs)
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
        super(AttentiveNodePooling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: nodes

            - nodes: Node features of shape (batch, [N], F)

        Returns:
            features: Hidden tensor of pooled edge attentions for each node.
        """
        node = inputs

        h = self.lay_pool_start(node)
        Wn = self.lay_linear_trafo(node)
        for _ in range(self.depth):
            hv = self.lay_gather_s([h, node])
            ev = self.lay_concat([hv, node])
            av = self.lay_alpha(ev)
            cont = self.lay_pool_attention([Wn, av])
            cont = self.lay_final_activ(cont)
            h,_ = self.lay_gru(cont, h, **kwargs)

        out = h
        return out

    def get_config(self):
        """Update layer config."""
        config = super(AttentiveNodePooling, self).get_config()
        config.update({"units": self.units, "depth": self.depth, "pooling_method": self.pooling_method})
        conf_sub = self.lay_alpha.get_config()
        for x in ["kernel_regularizer", "activity_regularizer", "bias_regularizer", "kernel_constraint",
                  "bias_constraint", "kernel_initializer", "bias_initializer", "activation", "use_bias"]:
            config.update({x: conf_sub[x]})
        conf_context = self.lay_final_activ.get_config()
        config.update({"activation_context": conf_context["activation"]})
        conf_gru = self.lay_gru.get_config()
        for x in ["recurrent_activation", "recurrent_initializer", "mrecurrent_regularizer", "recurrent_constraint",
                 "dropout", "recurrent_dropout", "reset_after"]:
            config.update({x: conf_gru[x]})
        return config

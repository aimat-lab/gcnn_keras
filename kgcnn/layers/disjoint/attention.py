import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.disjoint.gather import GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.utils.activ import kgcnn_custom_act
from kgcnn.utils.partition import _change_edge_tensor_indexing_by_row_partition
from kgcnn.utils.soft import segment_softmax


class PoolingLocalEdgesAttention(ks.layers.Layer):
    r"""
    Pooling all edges or edge-like features per node, corresponding to node assigned by edge indices.
    Uses attention for pooling. i.e.  $n_i =  \sum_j \alpha_{ij} e_ij $
    The attention is computed via: $\alpha_ij = softmax_j (a_ij)$ from the attention coefficients $a_ij$.
    The attention coefficients must be computed beforehand by edge features or by $\sigma( W n_i || W n_j)$ and
    are passed to this layer as input. Thereby this layer has no weights and only does pooling.
    In summary, $n_i =  \sum_j softmax_j(a_ij) e_ij $ is computed by the layer.

    If graphs indices were in 'sample' mode, the indices are corrected for disjoint graphs.
    Important: edge_index[:,0] are sorted for segment-operation.

    Args:
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
                             For disjoint representation 'batch' is default.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,
                 node_indexing="batch",
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 **kwargs):
        """Initialize layer."""
        super(PoolingLocalEdgesAttention, self).__init__(**kwargs)
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.partition_type = partition_type

    def build(self, input_shape):
        """Build layer."""
        super(PoolingLocalEdgesAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, node_partition, edges, attention, edge_partition, edge_indices]

            - node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
              only required for target shape, so that pooled tensor has same shape!
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edges (tf.tensor): Flatten edge features or node_j for edge_ij or node_i||node_j of shape (batch*None,F)
            - attention (tf.tensor): Attention coefficients to compute the attention from, must be shape (batch*None,1)
              and match the edges, i.e. have same first dimension and node assignment a(i,j) match e(i,j)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_indices (tf.tensor): Flatten index list tensor of shape (batch*None,2)
              pooling is done according to first index i from edge index pair (i,j)

        Returns:
            features (tf.tensor): Flatten feature tensor of pooled edge attentions for each node.
            The size will match the flatten node tensor.
            Output shape is (batch*None, F).
        """
        nod, node_part, edge, attention, edge_part, edgeind = inputs

        shiftind = _change_edge_tensor_indexing_by_row_partition(edgeind, node_part, edge_part,
                                                                 partition_type_node=self.partition_type,
                                                                 partition_type_edge=self.partition_type,
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
            pooled_index = tf.range(tf.shape(get)[0])  # tf.unique(nodind)
            outtarget_shape = (tf.shape(nod, out_type=nodind.dtype)[0], ks.backend.int_shape(dens)[-1])
            get = tf.scatter_nd(ks.backend.expand_dims(pooled_index, axis=-1), get, outtarget_shape)

        return get

    def get_config(self):
        """Update layer config."""
        config = super(PoolingLocalEdgesAttention, self).get_config()
        config.update({"is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type})
        return config


class AttentionHeadGAT(ks.layers.Layer):
    r"""Computes the attention head according to GAT.
    The attention coefficients are computed by $a_{ij} = \sigma( W n_i || W n_j)$,
    optionally by $a_{ij} = \sigma( W n_i || W n_j || e_{ij})$.
    The attention is obtained by $\alpha_ij = softmax_j (a_{ij})$.
    And the messages are pooled by $n_i =  \sum_j \alpha_{ij} e_ij $.
    And finally passed through an activation $h_i = \sigma(\sum_j \alpha_{ij} e_ij)$.

    If graphs indices were in 'sample' mode, the indices must be corrected for disjoint graphs.

    Args:
        units (int): Units for the linear trafo of node features before attention.
        activation (str): Activation. Default is {"class_name": "leaky_relu", "config": {"alpha": 0.2}},
            with fall-back "relu".
        use_bias (bool): Use bias. Default is True.
        kernel_regularizer: Kernel regularization. Default is None.
        bias_regularizer: Bias regularization. Default is None.
        activity_regularizer: Activity regularization. Default is None.
        kernel_constraint: Kernel constrains. Default is None.
        bias_constraint: Bias constrains. Default is None.
        kernel_initializer: Initializer for kernels. Default is 'glorot_uniform'.
        bias_initializer: Initializer for bias. Default is 'zeros'.
        use_edge_features (False): Append edge features to attention computation.
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
                             For disjoint representation 'batch' is default.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 use_edge_features=False,
                 node_indexing="batch",
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 **kwargs):
        """Initialize layer."""
        super(AttentionHeadGAT, self).__init__(**kwargs)
        # graph args
        self.is_sorted = is_sorted
        self.use_edge_features = use_edge_features
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.partition_type = partition_type

        # dense args
        self.units = int(units)
        if activation is None and "leaky_relu" in kgcnn_custom_act:
            activation = {"class_name": "leaky_relu", "config": {"alpha": 0.2}}
        elif activation is None:
            activation = "relu"
        self.use_bias = use_bias
        self.ath_activation = tf.keras.activations.get(activation)
        self.ath_kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.ath_bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.ath_kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.ath_bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.ath_activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.ath_kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.ath_bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.lay_linear_trafo = ks.layers.Dense(units, activation="linear", use_bias=use_bias,
                                                kernel_regularizer=kernel_regularizer,
                                                activity_regularizer=activity_regularizer,
                                                bias_regularizer=bias_regularizer,
                                                kernel_constraint=kernel_constraint,
                                                bias_constraint=bias_constraint,
                                                kernel_initializer=kernel_initializer,
                                                bias_initializer=bias_initializer)
        self.lay_alpha = ks.layers.Dense(1, activation=activation, use_bias=use_bias,
                                         kernel_regularizer=kernel_regularizer,
                                         activity_regularizer=activity_regularizer,
                                         bias_regularizer=bias_regularizer,
                                         kernel_constraint=kernel_constraint,
                                         bias_constraint=bias_constraint,
                                         kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer)
        self.lay_gather_in = GatherNodesIngoing(node_indexing=node_indexing, partition_type=partition_type)
        self.lay_gather_out = GatherNodesOutgoing(node_indexing=node_indexing, partition_type=partition_type)
        self.lay_concat = ks.layers.Concatenate(axis=-1)
        self.lay_pool_attention = PoolingLocalEdgesAttention(node_indexing=node_indexing, partition_type=partition_type,
                                                             has_unconnected=has_unconnected, is_sorted=is_sorted)
        self.lay_final_activ = ks.layers.Activation(activation=activation)

    def build(self, input_shape):
        """Build layer."""
        super(AttentionHeadGAT, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, node_partition, edges, edge_partition, edge_indices]

            - node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
              only required for target shape, so that pooled tensor has same shape!
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edges (tf.tensor): Flatten edge feature tensor of shape (batch*None,F)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_indices (tf.tensor): Flatten index list tensor of shape (batch*None,2)

        Returns:
            features (tf.tensor): Flatten feature tensor of pooled edge attentions for each node.
            The size will match the flatten node tensor.
            Output shape is (batch*None, F).
        """
        node, node_part, edge, edge_part, edge_index = inputs

        n_in = self.lay_gather_in([node, node_part, edge_index, edge_part])
        n_out = self.lay_gather_out([node, node_part, edge_index, edge_part])
        wn_in = self.lay_linear_trafo(n_in)
        wn_out = self.lay_linear_trafo(n_out)
        if self.use_edge_features:
            e_ij = self.lay_concat([wn_in, wn_out, edge])
        else:
            e_ij = self.lay_concat([wn_in, wn_out])
        a_ij = self.lay_alpha(e_ij)  # Should be dimension (batch*None,1)
        n_i = self.lay_pool_attention([node, node_part, wn_out, a_ij, edge_part, edge_index])
        out = self.lay_final_activ(n_i)

        return out

    def get_config(self):
        """Update layer config."""
        config = super(AttentionHeadGAT, self).get_config()
        config.update({"is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "use_edge_features": self.use_edge_features,
                       "units": self.units,
                       "use_bias": self.use_bias,
                       "activation": tf.keras.activations.serialize(self.ath_activation),
                       "kernel_regularizer": tf.keras.regularizers.serialize(self.ath_kernel_regularizer),
                       "bias_regularizer": tf.keras.regularizers.serialize(self.ath_bias_regularizer),
                       "activity_regularizer": tf.keras.regularizers.serialize(self.ath_activity_regularizer),
                       "kernel_constraint": tf.keras.constraints.serialize(self.ath_kernel_constraint),
                       "bias_constraint": tf.keras.constraints.serialize(self.ath_bias_constraint),
                       "kernel_initializer": tf.keras.initializers.serialize(self.ath_kernel_initializer),
                       "bias_initializer": tf.keras.initializers.serialize(self.ath_bias_initializer)
                       })
        return config

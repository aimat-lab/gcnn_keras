import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.layers.keras import Dense, Activation, Concatenate
from kgcnn.ops.activ import kgcnn_custom_act
from kgcnn.ops.casting import kgcnn_ops_cast_value_partition_to_ragged
from kgcnn.ops.casting import kgcnn_ops_dyn_cast
from kgcnn.ops.partition import kgcnn_ops_change_edge_tensor_indexing_by_row_partition
from kgcnn.ops.scatter import kgcnn_ops_scatter_segment_tensor_nd
from kgcnn.ops.segment import segment_softmax
from kgcnn.ops.types import kgcnn_ops_static_test_tensor_input_type, kgcnn_ops_check_tensor_type


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
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
    """

    def __init__(self,
                 node_indexing="sample",
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(PoolingLocalEdgesAttention, self).__init__(**kwargs)
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

    def build(self, input_shape):
        """Build layer."""
        super(PoolingLocalEdgesAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        The tensor representation can be tf.RaggedTensor, tf.Tensor or a list of (values, partition).
        The RaggedTensor has shape (batch, None, F) or in case of equal sized graphs (batch, N, F).
        For disjoint representation (values, partition), the node embeddings are given by
        a flatten value tensor of shape (batch*None, F) and a partition tensor of either "row_length",
        "row_splits" or "value_rowids" that matches the tf.RaggedTensor partition information. In this case
        the partition_type and node_indexing scheme, i.e. "batch", must be known by the layer.
        For edge indices, the last dimension holds indices from outgoing to ingoing node (i,j) as a directed edge.

        Args:
            inputs: [node, edges, attention, edge_indices]

            - nodes: Node features of shape (batch, [N], F)
            - edges: Edge or message features of shape (batch, [N], F)
            - attention: Attention coefficients of shape (batch, [N], 1)
            - edge_index: Edge indices of shape (batch, [N], F)

        Returns:
            embeddings: Feature tensor of pooled edge attentions for each node.
        """
        found_node_type = kgcnn_ops_check_tensor_type(inputs[0], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        found_edge_type = kgcnn_ops_check_tensor_type(inputs[1], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        found_att_type = kgcnn_ops_check_tensor_type(inputs[2], input_tensor_type=self.input_tensor_type,
                                                     node_indexing=self.node_indexing)
        found_index_type = kgcnn_ops_check_tensor_type(inputs[3], input_tensor_type=self.input_tensor_type,
                                                       node_indexing=self.node_indexing)

        # We cast to values here
        nod, node_part = kgcnn_ops_dyn_cast(inputs[0], input_tensor_type=found_node_type,
                                            output_tensor_type="values_partition",
                                            partition_type=self.partition_type)
        edge, _ = kgcnn_ops_dyn_cast(inputs[1], input_tensor_type=found_edge_type,
                                     output_tensor_type="values_partition",
                                     partition_type=self.partition_type)
        attention, _ = kgcnn_ops_dyn_cast(inputs[2], input_tensor_type=found_att_type,
                                          output_tensor_type="values_partition",
                                          partition_type=self.partition_type)
        edgeind, edge_part = kgcnn_ops_dyn_cast(inputs[3], input_tensor_type=found_index_type,
                                                output_tensor_type="values_partition",
                                                partition_type=self.partition_type)

        shiftind = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edgeind, node_part, edge_part,
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
            get = kgcnn_ops_scatter_segment_tensor_nd(get, nodind, tf.shape(nod))


        return kgcnn_ops_dyn_cast([get, node_part], input_tensor_type="values_partition",
                                  output_tensor_type=found_node_type, partition_type=self.partition_type)

    def get_config(self):
        """Update layer config."""
        config = super(PoolingLocalEdgesAttention, self).get_config()
        config.update({"is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
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
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
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
                 node_indexing="sample",
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(AttentionHeadGAT, self).__init__(**kwargs)
        # graph args
        self.is_sorted = is_sorted
        self.use_edge_features = use_edge_features
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

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

        kernel_args = {"use_bias": use_bias, "kernel_regularizer": kernel_regularizer,
                       "activity_regularizer": activity_regularizer, "bias_regularizer": bias_regularizer,
                       "kernel_constraint": kernel_constraint, "bias_constraint": bias_constraint,
                       "kernel_initializer": kernel_initializer, "bias_initializer": bias_initializer}
        dens_args = {"ragged_validate": self.ragged_validate, "input_tensor_type": self.input_tensor_type}
        dens_args.update(kernel_args)
        gather_args = {"input_tensor_type": self.input_tensor_type, "node_indexing": self.node_indexing}
        pooling_args = {"node_indexing": node_indexing, "partition_type": partition_type,
                        "has_unconnected": has_unconnected, "is_sorted": is_sorted,
                        "ragged_validate": self.ragged_validate, "input_tensor_type": self.input_tensor_type}

        self.lay_linear_trafo = Dense(units, activation="linear", **dens_args)
        self.lay_alpha = Dense(1, activation=activation, **dens_args)
        self.lay_gather_in = GatherNodesIngoing(**gather_args)
        self.lay_gather_out = GatherNodesOutgoing(**gather_args)
        self.lay_concat = Concatenate(axis=-1, input_tensor_type=self.input_tensor_type)
        self.lay_pool_attention = PoolingLocalEdgesAttention(**pooling_args)
        self.lay_final_activ = Activation(activation=activation, input_tensor_type=self.input_tensor_type)

    def build(self, input_shape):
        """Build layer."""
        super(AttentionHeadGAT, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        The tensor representation can be tf.RaggedTensor, tf.Tensor or a list of (values, partition).
        The RaggedTensor has shape (batch, None, F) or in case of equal sized graphs (batch, N, F).
        For disjoint representation (values, partition), the node embeddings are given by
        a flatten value tensor of shape (batch*None, F) and a partition tensor of either "row_length",
        "row_splits" or "value_rowids" that matches the tf.RaggedTensor partition information. In this case
        the partition_type and node_indexing scheme, i.e. "batch", must be known by the layer.
        For edge indices, the last dimension holds indices from outgoing to ingoing node (i,j) as a directed edge.

        Args:
            inputs (list): of [node, edges, edge_indices]

            - nodes: Node features of shape (batch, [N], F)
            - edges: Edge or message features of shape (batch, [N], F)
            - edge_index: Edge indices of shape (batch, [N], 2)

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

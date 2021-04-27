import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.ops.partition import kgcnn_ops_change_edge_tensor_indexing_by_row_partition, kgcnn_ops_change_partition_type
from kgcnn.ops.scatter import kgcnn_ops_scatter_segment_tensor_nd
from kgcnn.ops.segment import kgcnn_ops_segment_operation_by_name


class PoolingLocalEdges(ks.layers.Layer):
    """
    Pooling all edges or edge-like features per node, corresponding to node assigned by edge indices.
    
    If graphs indices were in 'sample' mode, the indices must be corrected for disjoint graphs.
    Apply e.g. segment_mean for index[0] incoming nodes. 
    Important: edge_index[:,0] are sorted for segment-operation.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'segment_mean'.
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
                             For disjoint representation 'batch' is default.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,
                 pooling_method="segment_mean",
                 node_indexing="batch",
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 **kwargs):
        """Initialize layer."""
        super(PoolingLocalEdges, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.partition_type = partition_type

    def build(self, input_shape):
        """Build layer."""
        super(PoolingLocalEdges, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, node_partition, edges, edge_partition, edge_indices]

            - node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
              only required for target shape and dtype.
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edges (tf.tensor): Flatten edge feature tensor of shape (batch*None,F)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_indices (tf.tensor): Flatten index list tensor of shape (batch*None,2)
              The index for segment reduction is taken from edge_indices[:,0].
    
        Returns:
            features (tf.tensor): Flatten feature tensor of pooled edge features for each node.
            The size will match the flatten node tensor.
            Output shape is (batch*None, F).
        """
        nod, node_part, edge, edge_part, edgeind = inputs

        shiftind = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edgeind, node_part, edge_part,
                                                                          partition_type_node=self.partition_type,
                                                                          partition_type_edge=self.partition_type,
                                                                          to_indexing='batch',
                                                                          from_indexing=self.node_indexing)

        nodind = shiftind[:, 0]  # Pick first index eg. ingoing
        dens = edge
        if not self.is_sorted:
            # Sort edgeindices
            node_order = tf.argsort(nodind, axis=0, direction='ASCENDING', stable=True)
            nodind = tf.gather(nodind, node_order, axis=0)
            dens = tf.gather(dens, node_order, axis=0)

        # Pooling via e.g. segment_sum
        out = kgcnn_ops_segment_operation_by_name(self.pooling_method, dens, nodind)

        if self.has_unconnected:
            out = kgcnn_ops_scatter_segment_tensor_nd(out, nodind, tf.shape(nod))

        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingLocalEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type})
        return config


PoolingLocalMessages = PoolingLocalEdges  # For now they are synonyms


class PoolingWeightedLocalEdges(ks.layers.Layer):
    """
    Pooling all edges or message/edge-like features per node, corresponding to node assigned by edge_indices.
    
    If graphs edge_indices were in 'sample' mode, the edge_indices must be corrected for disjoint graphs.
    Apply e.g. segment_mean for index[0] incoming nodes. 
    Important: edge_index[:,0] could be sorted for segment-operation.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'segment_mean'.
        is_sorted (bool): If the edge_indices are sorted for first ingoing index. Default is False.
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
                             For disjoint representation 'batch' is default.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,
                 pooling_method="segment_mean",
                 is_sorted=False,
                 node_indexing="batch",
                 has_unconnected=True,
                 normalize_by_weights=False,
                 partition_type="row_length",
                 **kwargs):
        """Initialize layer."""
        super(PoolingWeightedLocalEdges, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.node_indexing = node_indexing
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.normalize_by_weights = normalize_by_weights
        self.partition_type = partition_type

    def build(self, input_shape):
        """Build layer."""
        super(PoolingWeightedLocalEdges, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, node_partition, edges, edge_partition, edge_indices]

            - node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edges (tf.tensor): Flatten edge feature tensor of shape (batch*None,F)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_indices (tf.tensor): Flatten index list tensor of shape (batch*None,2)
              The index for segment reduction is taken from edge_indices[:,0] (ingoing node).
            - weights (tf.tensor): The weights could be the entry in the adjacency matrix for each edge in the list
              and must be broad-casted or match in dimension. Shape is e.g. (batch*None,1).
    
        Returns:
            features (tf.tensor): Flatten feature tensor of pooled edge features for each node.
            The size will match the flatten node tensor.
            Output shape is (batch*None, F).
        """
        nod, node_part, edge, edge_part, edgeind, weights = inputs

        shiftind = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edgeind, node_part, edge_part,
                                                                          partition_type_node=self.partition_type,
                                                                          partition_type_edge=self.partition_type,
                                                                          to_indexing='batch',
                                                                          from_indexing=self.node_indexing)

        wval = weights
        dens = edge * wval
        nodind = shiftind[:, 0]

        if not self.is_sorted:
            # Sort edgeindices
            node_order = tf.argsort(nodind, axis=0, direction='ASCENDING', stable=True)
            nodind = tf.gather(nodind, node_order, axis=0)
            dens = tf.gather(dens, node_order, axis=0)
            wval = tf.gather(wval, node_order, axis=0)

        # Pooling via e.g. segment_sum
        get = kgcnn_ops_segment_operation_by_name(self.pooling_method, dens, nodind)

        if self.normalize_by_weights:
            get = tf.math.divide_no_nan(get, tf.math.segment_sum(wval, nodind))  # +tf.eps

        if self.has_unconnected:
            get = kgcnn_ops_scatter_segment_tensor_nd(get, nodind, tf.shape(nod))

        out = get
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingWeightedLocalEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "normalize_by_weights": self.normalize_by_weights,
                       "partition_type": self.partition_type})
        return config


PoolingWeightedLocalMessages = PoolingWeightedLocalEdges  # For now they are synonyms


class PoolingNodes(ks.layers.Layer):
    """
    Polling all nodes per batch. The batch assignment is given by a length-tensor.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segment_function
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,
                 pooling_method="segment_mean",
                 partition_type="row_length",
                 **kwargs):
        """Initialize layer."""
        super(PoolingNodes, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.partition_type = partition_type

    def build(self, input_shape):
        """Build layer."""
        super(PoolingNodes, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [nodes, node_partition]

            - nodes (tf.tensor): Flatten node features of shape (batch*None,F)
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
    
        Returns:
            features (tf.tensor): Pooled node feature list of shape (batch,F)
            where F is the feature dimension and holds a pooled 
            node feature for each graph.
        """
        node, node_part = inputs

        batchi = kgcnn_ops_change_partition_type(node_part, self.partition_type, "value_rowids")

        out = kgcnn_ops_segment_operation_by_name(self.pooling_method, node, batchi)
        # Output should have correct shape
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingNodes, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "partition_type": self.partition_type})
        return config


class PoolingGlobalEdges(ks.layers.Layer):
    """
    Pooling all edges per graph. The batch assignment is given by a length-tensor.

    Args:
        pooling_method (str): Pooling method to use i.e. segement_function
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,
                 pooling_method="segment_mean",
                 partition_type="row_length",
                 **kwargs):
        """Initialize layer."""
        super(PoolingGlobalEdges, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.partition_type = partition_type

    def build(self, input_shape):
        """Build layer."""
        super(PoolingGlobalEdges, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [edges, edge_partition]

            - edges (tf.tensor): Flatten edge feature list of shape (batch*None,F)
            - edge_partition (tf.tensor): Row partition for edges. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
    
        Returns:
            features (tf.tensor): adj_matrix pooled edges feature list of shape (batch,F).
            where F is the feature dimension and holds a pooled 
            edge feature for each graph.
        """
        edge, edge_part = inputs

        batchi = kgcnn_ops_change_partition_type(edge_part, self.partition_type, "value_rowids")

        out = kgcnn_ops_segment_operation_by_name(self.pooling_method, edge, batchi)
        # Output already has correct shape
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingGlobalEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "partition_type": self.partition_type})
        return config


class PoolingLocalEdgesLSTM(ks.layers.Layer):
    """
    Pooling all edges or edge-like features per node, corresponding to node assigned by edge indices.
    Uses LSTM to aggregate Node-features.

    If graphs indices were in 'sample' mode, the indices must be corrected for disjoint graphs.
    Apply e.g. segment_mean for index[0] incoming nodes.
    Important: edge_index[:,0] are sorted for segment-operation.

    Args:
        units (int): Units for LSTM cell.
        pooling_method (str): Pooling method. Default is 'LSTM', is ignored.
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
                             For disjoint representation 'batch' is default.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,
                 units,
                 pooling_method="LSTM",
                 node_indexing="batch",
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 activation='tanh', recurrent_activation='sigmoid',
                 use_bias=True, kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros', unit_forget_bias=True,
                 kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
                 bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=False, return_state=False, go_backwards=False, stateful=False,
                 time_major=False, unroll=False,
                 **kwargs):
        """Initialize layer."""
        super(PoolingLocalEdgesLSTM, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.partition_type = partition_type

        self.lstm_unit = ks.layers.LSTM(units=units, activation=activation, recurrent_activation=recurrent_activation,
                                        use_bias=use_bias, kernel_initializer=kernel_initializer,
                                        recurrent_initializer=recurrent_initializer,
                                        bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias,
                                        kernel_regularizer=kernel_regularizer,
                                        recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer,
                                        activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                                        recurrent_constraint=recurrent_constraint,
                                        bias_constraint=bias_constraint, dropout=dropout,
                                        recurrent_dropout=recurrent_dropout,
                                        return_sequences=return_sequences, return_state=return_state,
                                        go_backwards=go_backwards, stateful=stateful,
                                        time_major=time_major, unroll=unroll)

    def build(self, input_shape):
        """Build layer."""
        super(PoolingLocalEdgesLSTM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, node_partition, edges, edge_partition, edge_indices]

            - node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edges (tf.tensor): Flatten edge feature tensor of shape (batch*None,F)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_indices (tf.tensor): Flatten index list tensor of shape (batch*None,2)
              The index for segment reduction is taken from edge_indices[:,0].

        Returns:
            features (tf.tensor): Flatten feature tensor of pooled edge features for each node.
            The size will match the flatten node tensor.
            Output shape is (batch*None, F).
        """
        nod, node_part, edge, edge_part, edgeind = inputs

        shiftind = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edgeind, node_part, edge_part,
                                                                          partition_type_node=self.partition_type,
                                                                          partition_type_edge=self.partition_type,
                                                                          to_indexing='batch',
                                                                          from_indexing=self.node_indexing)

        nodind = shiftind[:, 0]  # Pick first index eg. ingoing
        dens = edge
        if not self.is_sorted:
            # Sort edgeindices
            node_order = tf.argsort(nodind, axis=0, direction='ASCENDING', stable=True)
            nodind = tf.gather(nodind, node_order, axis=0)
            dens = tf.gather(dens, node_order, axis=0)

        # Pooling via LSTM
        # we make a ragged input
        ragged_lstm_input = tf.RaggedTensor.from_value_rowids(dens, nodind)
        get = self.lstm_unit(ragged_lstm_input)

        if self.has_unconnected:
            # Need to fill tensor since the maximum node may not be also in pooled
            # Does not happen if all nodes are also connected
            get = kgcnn_ops_scatter_segment_tensor_nd(get, nodind, tf.shape(nod))

        out = get
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingLocalEdgesLSTM, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type})
        conf_lstm = self.lstm_unit.get_config()
        lstm_param = ["activation", "recurrent_activation", "use_bias", "kernel_initializer", "recurrent_initializer",
                      "bias_initializer", "unit_forget_bias", "kernel_regularizer", "recurrent_regularizer",
                      "bias_regularizer", "activity_regularizer", "kernel_constraint", "recurrent_constraint",
                      "bias_constraint", "dropout", "recurrent_dropout", "implementation", "return_sequences",
                      "return_state", "go_backwards", "stateful", "time_major", "unroll"]
        for x in lstm_param:
            config.update({x: conf_lstm[x]})
        return config


PoolingLocalMessagesLSTM = PoolingLocalEdgesLSTM  # For now they are synonyms

import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.utils.partition import _change_edge_tensor_indexing_by_row_partition, _change_partition_type


class PoolingLocalEdges(ks.layers.Layer):
    """
    Pooling all edges or edgelike features per node, corresponding to node assigned by edge indices.
    
    If graphs indices were in 'sample' mode, the indices must be corrected for disjoint graphs.
    Apply e.g. segment_mean for index[0] incoming nodes. 
    Important: edge_index[:,0] are sorted for segment-operation.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segement_function. Default is 'segment_mean'.
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,
                 pooling_method="segment_mean",
                 node_indexing="batch",
                 is_sorted=True,
                 has_unconnected=False,
                 partition_type="row_length",
                 **kwargs):
        """Initialize layer."""
        super(PoolingLocalEdges, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.partition_type = partition_type

        if self.pooling_method == "segment_mean" or self.pooling_method == "mean":
            self._pool = tf.math.segment_mean
        elif self.pooling_method == "segment_sum" or self.pooling_method == "sum":
            self._pool = tf.math.segment_sum
        else:
            raise TypeError("Unknown pooling, choose: 'segment_mean', 'segment_sum', ...")

    def build(self, input_shape):
        """Build layer."""
        super(PoolingLocalEdges, self).build(input_shape)

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

        shiftind = _change_edge_tensor_indexing_by_row_partition(edgeind, node_part, edge_part,
                                                                 partition_type_node=self.partition_type,
                                                                 partition_type_edge=self.partition_type,
                                                                 to_indexing='batch',
                                                                 from_indexing=self.node_indexing)

        nodind = shiftind[:, 0]
        dens = edge
        if not self.is_sorted:
            # Sort edgeindices
            node_order = tf.argsort(nodind, axis=0, direction='ASCENDING', stable=True)
            nodind = tf.gather(nodind, node_order, axis=0)
            dens = tf.gather(dens, node_order, axis=0)

        # Pooling via e.g. segment_sum
        get = self._pool(dens, nodind)

        if self.has_unconnected:
            # Need to fill tensor since the maximum node may not be also in pooled
            # Does not happen if all nodes are also connected
            pooled_index = tf.range(tf.shape(get)[0])  # tf.unique(nodind)
            outtarget_shape = (tf.shape(nod, out_type=nodind.dtype)[0], ks.backend.int_shape(dens)[-1])
            get = tf.scatter_nd(ks.backend.expand_dims(pooled_index, axis=-1), get, outtarget_shape)

        out = get
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
    Pooling all edges or message/edgelike features per node, corresponding to node assigned by edge_indices.
    
    If graphs edge_indices were in 'sample' mode, the edge_indices must be corrected for disjoint graphs.
    Apply e.g. segment_mean for index[0] incoming nodes. 
    Important: edge_index[:,0] could be sorted for segment-operation.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segement_function. Default is 'segment_mean'.
        is_sorted (bool): If the edge_indices are sorted for first ingoing index. Default is False.
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,
                 pooling_method="segment_mean",
                 is_sorted=True,
                 node_indexing="batch",
                 has_unconnected=False,
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

        if self.pooling_method == "segment_mean":
            self._pool = tf.math.segment_mean
        elif self.pooling_method == "segment_sum":
            self._pool = tf.math.segment_sum
        else:
            raise TypeError("Unknown pooling, choose: 'segment_mean', 'segment_sum', ...")

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
            - weights (tf.tensor): The weights could be the entry in the ajacency matrix for each edge in the list
              and must be broadcasted or match in dimension. Shape is e.g. (batch*None,1).
    
        Returns:
            features (tf.tensor): Flatten feature tensor of pooled edge features for each node.
            The size will match the flatten node tensor.
            Output shape is (batch*None, F).
        """
        nod, node_part, edge, edge_part, edgeind, weights = inputs

        shiftind = _change_edge_tensor_indexing_by_row_partition(edgeind, node_part, edge_part,
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
        get = self._pool(dens, nodind)

        if self.normalize_by_weights:
            get = tf.math.divide_no_nan(get, tf.math.segment_sum(wval, nodind))  # +tf.eps

        if self.has_unconnected:
            # Need to fill tensor since the maximum node may not be also in pooled
            # Does not happen if all nodes are also connected
            pooled_index = tf.range(tf.shape(get)[0])  # tf.unique(nodind)
            outtarget_shape = (tf.shape(nod, out_type=nodind.dtype)[0], ks.backend.int_shape(dens)[-1])
            get = tf.scatter_nd(ks.backend.expand_dims(pooled_index, axis=-1), get, outtarget_shape)

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
        pooling_method (str): Pooling method to use i.e. segement_function
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

        if self.pooling_method == "segment_mean":
            self._pool = tf.math.segment_mean
        elif self.pooling_method == "segment_sum":
            self._pool = tf.math.segment_sum
        else:
            raise TypeError("Unknown pooling, choose: 'segment_mean', 'segment_sum', ...")

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

        batchi = _change_partition_type(node_part, self.partition_type, "value_rowids")

        out = self._pool(node, batchi)
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

        if self.pooling_method == "segment_mean":
            self._pool = tf.math.segment_mean
        elif self.pooling_method == "segment_sum":
            self._pool = tf.math.segment_sum
        else:
            raise TypeError("Unknown pooling, choose: 'segment_mean', 'segment_sum', ...")

    def build(self, input_shape):
        """Build layer."""
        super(PoolingGlobalEdges, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [egdes, edge_partition]

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

        batchi = _change_partition_type(edge_part, self.partition_type, "value_rowids")

        out = self._pool(edge, batchi)
        # Output already has correct shape
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingGlobalEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "partition_type": self.partition_type})
        return config

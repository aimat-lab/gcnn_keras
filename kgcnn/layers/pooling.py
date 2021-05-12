import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.ops.casting import kgcnn_ops_dyn_cast
from kgcnn.ops.partition import kgcnn_ops_change_edge_tensor_indexing_by_row_partition, kgcnn_ops_change_partition_type
from kgcnn.ops.scatter import kgcnn_ops_scatter_segment_tensor_nd
from kgcnn.ops.segment import kgcnn_ops_segment_operation_by_name
from kgcnn.ops.types import kgcnn_ops_static_test_tensor_input_type, kgcnn_ops_check_tensor_type


class PoolingLocalEdges(ks.layers.Layer):
    """
    Pooling all edges or edge-like features per node, corresponding to node assigned by edge indices.
    
    If graphs indices were in 'sample' mode, the indices must be corrected for disjoint graphs.
    Apply e.g. segment_mean for index[0] incoming nodes. 
    Important: edge_index[:,0] are sorted for segment-operation.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
    """

    def __init__(self,
                 pooling_method="mean",
                 node_indexing="sample",
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(PoolingLocalEdges, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]
        self._supports_ragged_inputs = True

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

    def build(self, input_shape):
        """Build layer."""
        super(PoolingLocalEdges, self).build(input_shape)

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
            inputs (list): of [node, edges, edge_index]

            - nodes: Node features of shape (batch, [N], F)
            - edges: Edge or message features of shape (batch, [N], F)
            - edge_index: Edge indices of shape (batch, [N], 2)
    
        Returns:
            features: Pooled feature tensor of pooled edge features for each node.
        """
        found_node_type = kgcnn_ops_check_tensor_type(inputs[0], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        found_edge_type = kgcnn_ops_check_tensor_type(inputs[1], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        found_index_type = kgcnn_ops_check_tensor_type(inputs[2], input_tensor_type=self.input_tensor_type,
                                                       node_indexing=self.node_indexing)

        nod, node_part = kgcnn_ops_dyn_cast(inputs[0], input_tensor_type=found_node_type,
                                            output_tensor_type="values_partition",
                                            partition_type=self.partition_type)
        edge, _ = kgcnn_ops_dyn_cast(inputs[1], input_tensor_type=found_edge_type,
                                     output_tensor_type="values_partition",
                                     partition_type=self.partition_type)
        edgeind, edge_part = kgcnn_ops_dyn_cast(inputs[2], input_tensor_type=found_index_type,
                                                output_tensor_type="values_partition",
                                                partition_type=self.partition_type)

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

        return kgcnn_ops_dyn_cast([out, node_part], input_tensor_type="values_partition",
                                  output_tensor_type=found_node_type, partition_type=self.partition_type)

    def get_config(self):
        """Update layer config."""
        config = super(PoolingLocalEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate
                       })
        return config


PoolingLocalMessages = PoolingLocalEdges  # For now they are synonyms


class PoolingWeightedLocalEdges(ks.layers.Layer):
    """
    Pooling all edges or message/edge-like features per node, corresponding to node assigned by edge_indices.
    
    If graphs edge_indices were in 'sample' mode, the edge_indices must be corrected for disjoint graphs.
    Apply e.g. segment_mean for index[0] incoming nodes. 
    Important: edge_index[:,0] could be sorted for segment-operation.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
        is_sorted (bool): If the edge_indices are sorted for first ingoing index. Default is False.
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        normalize_by_weights (bool): Normalize the pooled output by the sum of weights. Default is False.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
        **kwargs
    """

    def __init__(self,
                 pooling_method="mean",
                 is_sorted=False,
                 node_indexing="sample",
                 has_unconnected=True,
                 normalize_by_weights=False,
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(PoolingWeightedLocalEdges, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.node_indexing = node_indexing
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.normalize_by_weights = normalize_by_weights
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]
        self._supports_ragged_inputs = True

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

    def build(self, input_shape):
        """Build layer."""
        super(PoolingWeightedLocalEdges, self).build(input_shape)

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
            inputs (list): of [node, edges, edge_index, weights]

            - nodes: Node features of shape (batch, [N], F)
            - edges: Edge or message features of shape (batch, [N], F)
            - edge_index: Edge indices of shape (batch, [N], 2)
            - weights: Edge or message weights. Most broadcast to edges or messages, e.g. (batch, [N], 1)

        Returns:
            features: Pooled feature tensor of pooled edge features for each node.
        """
        found_node_type = kgcnn_ops_check_tensor_type(inputs[0], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        found_edge_type = kgcnn_ops_check_tensor_type(inputs[1], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        found_index_type = kgcnn_ops_check_tensor_type(inputs[2], input_tensor_type=self.input_tensor_type,
                                                       node_indexing=self.node_indexing)
        found_weight_type = kgcnn_ops_check_tensor_type(inputs[3], input_tensor_type=self.input_tensor_type,
                                                        node_indexing=self.node_indexing)

        nod, node_part = kgcnn_ops_dyn_cast(inputs[0], input_tensor_type=found_node_type,
                                            output_tensor_type="values_partition",
                                            partition_type=self.partition_type)
        edge, _ = kgcnn_ops_dyn_cast(inputs[1], input_tensor_type=found_edge_type,
                                     output_tensor_type="values_partition",
                                     partition_type=self.partition_type)
        edgeind, edge_part = kgcnn_ops_dyn_cast(inputs[2], input_tensor_type=found_index_type,
                                                output_tensor_type="values_partition",
                                                partition_type=self.partition_type)
        weights, _ = kgcnn_ops_dyn_cast(inputs[4], input_tensor_type=found_weight_type,
                                        output_tensor_type="values_partition",
                                        partition_type=self.partition_type)

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

        return kgcnn_ops_dyn_cast([get, node_part], input_tensor_type="values_partition",
                                  output_tensor_type=found_node_type, partition_type=self.partition_type)

    def get_config(self):
        """Update layer config."""
        config = super(PoolingWeightedLocalEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "normalize_by_weights": self.normalize_by_weights,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
        return config


PoolingWeightedLocalMessages = PoolingWeightedLocalEdges  # For now they are synonyms


class PoolingNodes(ks.layers.Layer):
    """
    Polling all nodes per batch. The batch assignment is given by a length-tensor.
    
    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
        is_sorted (bool): If the edge_indices are sorted for first ingoing index. Default is False.
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
    """

    def __init__(self,
                 pooling_method="mean",
                 is_sorted=False,
                 node_indexing="sample",
                 has_unconnected=True,
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(PoolingNodes, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self.node_indexing = node_indexing
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

        self._supports_ragged_inputs = True

    def build(self, input_shape):
        """Build layer."""
        super(PoolingNodes, self).build(input_shape)

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
            inputs: Node features of shape (batch, [N], F)
    
        Returns:
            nodes (tf.tensor): Pooled node features of shape (batch,F)
        """
        found_node_type = kgcnn_ops_check_tensor_type(inputs, input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        nod, node_part = kgcnn_ops_dyn_cast(inputs, input_tensor_type=found_node_type,
                                            output_tensor_type="values_partition",
                                            partition_type=self.partition_type)

        batchi = kgcnn_ops_change_partition_type(node_part, self.partition_type, "value_rowids")

        out = kgcnn_ops_segment_operation_by_name(self.pooling_method, nod, batchi)

        # Output should have correct shape
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingNodes, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
        return config


class PoolingWeightedNodes(ks.layers.Layer):
    """
    Polling all nodes per batch. The batch assignment is given by a length-tensor.

    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
        is_sorted (bool): If the edge_indices are sorted for first ingoing index. Default is False.
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
    """

    def __init__(self,
                 pooling_method="mean",
                 is_sorted=False,
                 node_indexing="sample",
                 has_unconnected=True,
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(PoolingWeightedNodes, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self.node_indexing = node_indexing
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

        self._supports_ragged_inputs = True

    def build(self, input_shape):
        """Build layer."""
        super(PoolingWeightedNodes, self).build(input_shape)

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
            inputs (list): of [node, weights]

            - nodes: Node features of shape (batch, [N], F)
            - weights: Edge or message weights. Most broadcast to nodes.

        Returns:
            nodes (tf.tensor): Pooled node features of shape (batch,F)
        """
        found_node_type = kgcnn_ops_check_tensor_type(inputs[0], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        found_weight_type = kgcnn_ops_check_tensor_type(inputs[1], input_tensor_type=self.input_tensor_type,
                                                        node_indexing=self.node_indexing)
        nod, node_part = kgcnn_ops_dyn_cast(inputs[0], input_tensor_type=found_node_type,
                                            output_tensor_type="values_partition",
                                            partition_type=self.partition_type)
        weights, _ = kgcnn_ops_dyn_cast(inputs[1], input_tensor_type=found_weight_type,
                                        output_tensor_type="values_partition",
                                        partition_type=self.partition_type)

        batchi = kgcnn_ops_change_partition_type(node_part, self.partition_type, "value_rowids")

        nod = tf.math.multiply(nod, weights)
        out = kgcnn_ops_segment_operation_by_name(self.pooling_method, nod, batchi)
        # Output should have correct shape
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingWeightedNodes, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
        return config


class PoolingGlobalEdges(ks.layers.Layer):
    """
    Pooling all edges per graph. The batch assignment is given by a length-tensor.

    Args:
        pooling_method (str): Pooling method to use i.e. segment_function. Default is 'mean'.
        is_sorted (bool): If the edge_indices are sorted for first ingoing index. Default is False.
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
    """

    def __init__(self,
                 pooling_method="mean",
                 is_sorted=False,
                 node_indexing="sample",
                 has_unconnected=True,
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(PoolingGlobalEdges, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self.node_indexing = node_indexing
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

        self._supports_ragged_inputs = True

    def build(self, input_shape):
        """Build layer."""
        super(PoolingGlobalEdges, self).build(input_shape)

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
            inputs: Edge features or message embeddings of shape (batch, [N], F)
    
        Returns:
            tf.tensor: Pooled edges feature list of shape (batch,F).
        """
        found_edge_type = kgcnn_ops_check_tensor_type(inputs, input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        edge, edge_part = kgcnn_ops_dyn_cast(inputs, input_tensor_type=found_edge_type,
                                             output_tensor_type="values_partition",
                                             partition_type=self.partition_type)

        batchi = kgcnn_ops_change_partition_type(edge_part, self.partition_type, "value_rowids")

        out = kgcnn_ops_segment_operation_by_name(self.pooling_method, edge, batchi)
        # Output already has correct shape and type
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingGlobalEdges, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
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
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`). If you pass `None`, no activation
            is applied (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step.
            Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
            applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean (default `True`), whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix, used for
            the linear transformation of the inputs. Default: `glorot_uniform`.
            recurrent_initializer: Initializer for the `recurrent_kernel` weights
            matrix, used for the linear transformation of the recurrent state.
            Default: `orthogonal`.
        bias_initializer: Initializer for the bias vector. Default: `zeros`.
            unit_forget_bias: Boolean (default `True`). If True, add 1 to the bias of
            the forget gate at initialization. Setting it to true will also force
            `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
            al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
            bias_regularizer: Regularizer function applied to the bias vector. Default:
            `None`.
        activity_regularizer: Regularizer function applied to the output of the
            layer (its "activation"). Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the `recurrent_kernel`
            weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector. Default:
            `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the linear
            transformation of the inputs. Default: 0.
            recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
            the linear transformation of the recurrent state. Default: 0.
        return_sequences: Boolean. Whether to return the last output. in the output
            sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state in addition to the
            output. Default: `False`.
        go_backwards: Boolean (default `False`). If True, process the input sequence
            backwards and return the reversed sequence.
        stateful: Boolean (default `False`). If True, the last state for each sample
            at index i in a batch will be used as initial state for the sample of
            index i in the following batch.
        time_major: The shape format of the `inputs` and `outputs` tensors.
            If True, the inputs and outputs will be in shape
            `[timesteps, batch, feature]`, whereas in the False case, it will be
            `[batch, timesteps, feature]`. Using `time_major = True` is a bit more
            efficient because it avoids transposes at the beginning and end of the
            RNN calculation. However, most TensorFlow data is batch-major, so by
            default this function accepts input and emits output in batch-major
            form.
        unroll: Boolean (default `False`). If True, the network will be unrolled,
            else a symbolic loop will be used. Unrolling can speed-up a RNN, although
            it tends to be more memory-intensive. Unrolling is only suitable for short
            sequences.
    """

    def __init__(self,
                 units,
                 pooling_method="LSTM",
                 node_indexing="sample",
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
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
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self._supports_ragged_inputs = True
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

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
        if self.pooling_method not in ["LSTM", "lstm"]:
            print("Warning: Pooling method does not match with layer, expected 'LSTM' but got", self.pooling_method)

    def build(self, input_shape):
        """Build layer."""
        super(PoolingLocalEdgesLSTM, self).build(input_shape)

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
            inputs (list): [nodes, edges, edge_index]

            - nodes: Node features of shape (batch, [N], F)
            - edges: Edge or message features of shape (batch, [N], F)
            - edge_index: Edge indices of shape (batch, [N], 2)


        Returns:
            features: Feature tensor of pooled edge features for each node.
        """
        found_node_type = kgcnn_ops_check_tensor_type(inputs[0], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        found_edge_type = kgcnn_ops_check_tensor_type(inputs[1], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        found_index_type = kgcnn_ops_check_tensor_type(inputs[2], input_tensor_type=self.input_tensor_type,
                                                       node_indexing=self.node_indexing)

        nod, node_part = kgcnn_ops_dyn_cast(inputs[0], input_tensor_type=found_node_type,
                                            output_tensor_type="values_partition",
                                            partition_type=self.partition_type)
        edge, _ = kgcnn_ops_dyn_cast(inputs[1], input_tensor_type=found_edge_type,
                                     output_tensor_type="values_partition",
                                     partition_type=self.partition_type)
        edgeind, edge_part = kgcnn_ops_dyn_cast(inputs[2], input_tensor_type=found_index_type,
                                                output_tensor_type="values_partition",
                                                partition_type=self.partition_type)

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

        return kgcnn_ops_dyn_cast([get, node_part], input_tensor_type="values_partition",
                                  output_tensor_type=found_node_type, partition_type=self.partition_type)

    def get_config(self):
        """Update layer config."""
        config = super(PoolingLocalEdgesLSTM, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "ragged_validate": self.ragged_validate})
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

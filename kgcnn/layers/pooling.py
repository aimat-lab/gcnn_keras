import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.ops.partition import kgcnn_ops_change_edge_tensor_indexing_by_row_partition, kgcnn_ops_change_partition_type
from kgcnn.ops.scatter import kgcnn_ops_scatter_segment_tensor_nd
from kgcnn.ops.segment import kgcnn_ops_segment_operation_by_name
from kgcnn.ops.casting import kgcnn_ops_cast_ragged_to_value_partition, kgcnn_ops_cast_value_partition_to_ragged


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
        **kwargs
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
        self._tensor_input_type_implemented = ["ragged", "values_partition"]
        self._supports_ragged_inputs = True

        if self.input_tensor_type not in self._tensor_input_type_implemented:
            raise NotImplementedError("Error: Tensor input type ", self.input_tensor_type,
                                      "is not implemented for this layer ", self.name, "choose one of the following:",
                                      self._tensor_input_type_implemented)
        if self.input_tensor_type == "ragged" and self.node_indexing != "sample":
            print("Warning: For ragged tensor input, default node_indexing is considered 'sample'. ")
        if self.input_tensor_type == "values_partition" and self.node_indexing != "batch":
            print("Warning: For values_partition tensor input, default node_indexing is considered 'batch'. ")

    def build(self, input_shape):
        """Build layer."""
        super(PoolingLocalEdges, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, edges, edge_index]

            - nodes: Node features.
              This can be either a tuple of (values, partition) tensors of shape (batch*None,F)
              and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
              disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of shape (batch, N, F)
              and mask (batch, N) or a single RaggedTensor of shape (batch,None,F)
              or a singe tensor for equally sized graphs (batch,N,F).
            - edges: Edge or message features.
              This can be either a tuple of (values, partition) tensors of shape (batch*None,F)
              and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
              disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of shape (batch, N, F)
              and mask (batch, N) or a single RaggedTensor of shape (batch,None,F)
              or a singe tensor for equally sized graphs (batch,N,F).
            - edge_index: Edge indices.
              This can be either a tuple of (values, partition) tensors of shape (batch*None,2)
              and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
              disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of shape (batch, N, 2)
              and mask (batch, N) or a single RaggedTensor of shape (batch,None,2)
              or a singe tensor for equally sized graphs (batch,N,2).
    
        Returns:
            features: Pooled feature tensor of pooled edge features for each node.
        """
        nod, node_part, edge, _, edgeind, edge_part = None, None, None, None, None, None

        if self.input_tensor_type == "values_partition":
            [nod, node_part], [edge, _], [edgeind, edge_part] = inputs
        elif self.input_tensor_type == "ragged":
            nod, node_part = kgcnn_ops_cast_ragged_to_value_partition(inputs[0], self.partition_type)
            edge, _ = kgcnn_ops_cast_ragged_to_value_partition(inputs[1], self.partition_type)
            edgeind, edge_part = kgcnn_ops_cast_ragged_to_value_partition(inputs[2],  self.partition_type)

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

        if self.input_tensor_type == "values_partition":
            return [out, node_part]
        elif self.input_tensor_type == "ragged":
            return kgcnn_ops_cast_value_partition_to_ragged([out, node_part], self.partition_type)

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
        self._tensor_input_type_implemented = ["ragged", "values_partition"]
        self._supports_ragged_inputs = True

        if self.input_tensor_type not in self._tensor_input_type_implemented:
            raise NotImplementedError("Error: Tensor input type ", self.input_tensor_type,
                                      "is not implemented for this layer ", self.name, "choose one of the following:",
                                      self._tensor_input_type_implemented)
        if self.input_tensor_type == "ragged" and self.node_indexing != "sample":
            print("Warning: For ragged tensor input, default node_indexing is considered 'sample'. ")
        if self.input_tensor_type == "values_partition" and self.node_indexing != "batch":
            print("Warning: For values_partition tensor input, default node_indexing is considered 'batch'. ")

    def build(self, input_shape):
        """Build layer."""
        super(PoolingWeightedLocalEdges, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, edges, edge_index, weights]

            - nodes: Node features.
              This can be either a tuple of (values, partition) tensors of shape (batch*None,F)
              and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
              disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of shape (batch, N, F)
              and mask (batch, N) or a single RaggedTensor of shape (batch,None,F)
              or a singe tensor for equally sized graphs (batch,N,F).
            - edges: Edge or message features.
              This can be either a tuple of (values, partition) tensors of shape (batch*None,F)
              and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
              disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of shape (batch, N, F)
              and mask (batch, N) or a single RaggedTensor of shape (batch,None,F)
              or a singe tensor for equally sized graphs (batch,N,F).
            - edge_index: Edge indices.
              This can be either a tuple of (values, partition) tensors of shape (batch*None,2)
              and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
              disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of shape (batch, N, 2)
              and mask (batch, N) or a single RaggedTensor of shape (batch,None,2)
              or a singe tensor for equally sized graphs (batch,N,2).
            - weights: Edge or message weights. Most broadcast to edges or messages.
              This can be either a tuple of (values, partition) tensors of shape (batch*None,1)
              and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
              disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of shape (batch, N, 1)
              and mask (batch, N) or a single RaggedTensor of shape (batch,None,1)
              or a singe tensor for equally sized graphs (batch,N,1).

        Returns:
            features: Pooled feature tensor of pooled edge features for each node.
        """
        nod, node_part, edge, _, edgeind, edge_part, weights, _ = None, None, None, None, None, None, None, None

        if self.input_tensor_type == "values_partition":
            [nod, node_part], [edge, _], [edgeind, edge_part], [weights, _] = inputs
        elif self.input_tensor_type == "ragged":
            nod, node_part = kgcnn_ops_cast_ragged_to_value_partition(inputs[0], self.partition_type)
            edge, _ = kgcnn_ops_cast_ragged_to_value_partition(inputs[1], self.partition_type)
            edgeind, edge_part = kgcnn_ops_cast_ragged_to_value_partition(inputs[2], self.partition_type)
            weights, _ = kgcnn_ops_cast_ragged_to_value_partition(inputs[3], self.partition_type)

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

        if self.input_tensor_type == "values_partition":
            return [get, node_part]
        elif self.input_tensor_type == "ragged":
            return kgcnn_ops_cast_value_partition_to_ragged([get, node_part], self.partition_type)

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
        **kwargs
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
        self._tensor_input_type_implemented = ["ragged", "values_partition"]
        if self.input_tensor_type not in self._tensor_input_type_implemented:
            raise NotImplementedError("Error: Tensor input type ", self.input_tensor_type,
                                      "is not implemented for this layer ", self.name, "choose one of the following:",
                                      self._tensor_input_type_implemented)

        self._supports_ragged_inputs = True

    def build(self, input_shape):
        """Build layer."""
        super(PoolingNodes, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: Node features.
                This can be either a tuple of (values, partition) tensors of shape (batch*None,F)
                and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
                disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of shape
                (batch, N, F) and mask (batch, N) or a single RaggedTensor of shape (batch,None,F)
                or a singe tensor for equally sized graphs (batch,N,F).
    
        Returns:
            nodes (tf.tensor): Pooled node features of shape (batch,F)
        """
        nod, node_part = None, None
        if self.input_tensor_type == "values_partition":
            nod, node_part = inputs
        elif self.input_tensor_type == "ragged":
            nod, node_part = kgcnn_ops_cast_ragged_to_value_partition(inputs, self.partition_type)

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
        **kwargs
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
        self._tensor_input_type_implemented = ["ragged", "values_partition"]
        if self.input_tensor_type not in self._tensor_input_type_implemented:
            raise NotImplementedError("Error: Tensor input type ", self.input_tensor_type,
                                      "is not implemented for this layer ", self.name, "choose one of the following:",
                                      self._tensor_input_type_implemented)

        self._supports_ragged_inputs = True

    def build(self, input_shape):
        """Build layer."""
        super(PoolingWeightedNodes, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, weights]

            - nodes: Node features.
              This can be either a tuple of (values, partition) tensors of shape (batch*None,F)
              and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
              disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of shape (batch, N, F)
              and mask (batch, N) or a single RaggedTensor of shape (batch,None,F)
              or a singe tensor for equally sized graphs (batch,N,F).
            - weights: Edge or message weights. Most broadcast to nodes.
              This can be either a tuple of (values, partition) tensors of shape (batch*None,1)
              and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
              disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of shape (batch, N, 1)
              and mask (batch, N) or a single RaggedTensor of shape (batch,None,1)
              or a singe tensor for equally sized graphs (batch,N,1).

        Returns:
            nodes (tf.tensor): Pooled node features of shape (batch,F)
        """
        nod, node_part, weights = None, None, None
        if self.input_tensor_type == "values_partition":
            [nod, node_part], [weights, _] = inputs
        elif self.input_tensor_type == "ragged":
            nod, node_part = kgcnn_ops_cast_ragged_to_value_partition(inputs, self.partition_type)
            weights, _ = kgcnn_ops_cast_ragged_to_value_partition(inputs[1], self.partition_type)

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
        **kwargs
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
        self._tensor_input_type_implemented = ["ragged", "values_partition"]
        if self.input_tensor_type not in self._tensor_input_type_implemented:
            raise NotImplementedError("Error: Tensor input type ", self.input_tensor_type,
                                      "is not implemented for this layer ", self.name, "choose one of the following:",
                                      self._tensor_input_type_implemented)

        self._supports_ragged_inputs = True

    def build(self, input_shape):
        """Build layer."""
        super(PoolingGlobalEdges, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: Edge features or message embeddings.
                This can be either a tuple of (values, partition) tensors of shape (batch*None,F)
                and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
                disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of
                shape (batch, N, F) and mask (batch, N) or a single RaggedTensor of shape (batch,None,F)
                or a singe tensor for equally sized graphs (batch,N,F).
    
        Returns:
            tf.tensor: Pooled edges feature list of shape (batch,F).
        """
        edge, edge_part = None, None
        if self.input_tensor_type == "values_partition":
            edge, edge_part = inputs
        elif self.input_tensor_type == "ragged":
            edge, edge_part = kgcnn_ops_cast_ragged_to_value_partition(inputs, self.partition_type)

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
        **kwargs
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
        self._tensor_input_type_implemented = ["ragged", "values_partition"]
        if self.input_tensor_type not in self._tensor_input_type_implemented:
            raise NotImplementedError("Error: Tensor input type ", self.input_tensor_type,
                                      "is not implemented for this layer ", self.name, "choose one of the following:",
                                      self._tensor_input_type_implemented)

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
            inputs (list): [nodes, edges, edge_index]

            - nodes: Node features.
              This can be either a tuple of (values, partition) tensors of shape (batch*None,F)
              and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
              disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of shape (batch, N, F)
              and mask (batch, N) or a single RaggedTensor of shape (batch,None,F)
              or a singe tensor for equally sized graphs (batch,N,F).
            - edges: Edge or message features.
              This can be either a tuple of (values, partition) tensors of shape (batch*None,F)
              and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
              disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of shape (batch, N, F)
              and mask (batch, N) or a single RaggedTensor of shape (batch,None,F)
              or a singe tensor for equally sized graphs (batch,N,F).
            - edge_index: Edge indices.
              This can be either a tuple of (values, partition) tensors of shape (batch*None,2)
              and a partition tensor of the type "row_length", "row_splits" or "value_rowids". This usually uses
              disjoint indexing defined by 'node_indexing'. Or a tuple of (values, mask) tensors of shape (batch, N, 2)
              and mask (batch, N) or a single RaggedTensor of shape (batch,None,2)
              or a singe tensor for equally sized graphs (batch,N,2).

        Returns:
            features: Feature tensor of pooled edge features for each node.
        """
        nod, node_part, edge, _, edgeind, edge_part = None, None, None, None, None, None
        if self.input_tensor_type == "values_partition":
            [nod, node_part], [edge, _], [edgeind, edge_part] = inputs
        elif self.input_tensor_type == "ragged":
            nod, node_part = kgcnn_ops_cast_ragged_to_value_partition(inputs[0], self.partition_type)
            edge, _ = kgcnn_ops_cast_ragged_to_value_partition(inputs[1], self.partition_type)
            edgeind, edge_part = kgcnn_ops_cast_ragged_to_value_partition(inputs[2],  self.partition_type)

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


        if self.input_tensor_type == "values_partition":
            return [get, node_part]
        elif self.input_tensor_type == "ragged":
            return kgcnn_ops_cast_value_partition_to_ragged([get, node_part], self.partition_type)


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

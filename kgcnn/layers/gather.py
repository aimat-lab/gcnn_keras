import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.casting import kgcnn_ops_dyn_cast
from kgcnn.ops.partition import kgcnn_ops_change_partition_type, kgcnn_ops_change_edge_tensor_indexing_by_row_partition
from kgcnn.ops.types import kgcnn_ops_static_test_tensor_input_type, kgcnn_ops_check_tensor_type


class GatherNodes(ks.layers.Layer):
    """
    Gather nodes by node indices. Index list must match node tensor. An edge is defined by index tuple (i,j).
    
    If graphs edge_indices were in 'sample' mode, the edge_indices must be corrected for disjoint graphs.
    
    Args:
        concat_nodes (bool): Whether to concatenate gathered node features. Default is True.
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is expected.
        partition_type (str): Partition tensor type to assign nodes or edges to batch. Default is "row_length".
            This is used for input_tensor_type="values_partition".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor output. Default is False.
        is_sorted (bool): Whether edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): Whether unconnected nodes are allowed. Default is True.
    """

    def __init__(self,
                 concat_nodes=True,
                 node_indexing='sample',
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 is_sorted=False,
                 has_unconnected=True,
                 **kwargs):
        """Initialize layer."""
        super(GatherNodes, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.concat_nodes = concat_nodes
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]
        self._supports_ragged_inputs = True

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

    def build(self, input_shape):
        """Build layer."""
        super(GatherNodes, self).build(input_shape)

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
            inputs (list): [nodes, edge_index]

            - nodes: Node embeddings of shape (batch, [N], F)
            - edge_index: Edge indices of shape (batch, [M], 2)
            
        Returns:
            embeddings: Gathered node embeddings that match the number of edges.
        """
        found_node_type = kgcnn_ops_check_tensor_type(inputs[0], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        found_edge_type = kgcnn_ops_check_tensor_type(inputs[1], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)

        # We cast to values here
        node, node_part = kgcnn_ops_dyn_cast(inputs[0], input_tensor_type=found_node_type,
                                             output_tensor_type="values_partition",
                                             partition_type=self.partition_type)
        edge_index, edge_part = kgcnn_ops_dyn_cast(inputs[1], input_tensor_type=found_edge_type,
                                                   output_tensor_type="values_partition",
                                                   partition_type=self.partition_type)

        indexlist = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edge_index, node_part, edge_part,
                                                                           partition_type_node=self.partition_type,
                                                                           partition_type_edge=self.partition_type,
                                                                           to_indexing='batch',
                                                                           from_indexing=self.node_indexing)
        out = tf.gather(node, indexlist, axis=0)
        if self.concat_nodes:
            out = tf.keras.backend.concatenate([out[:, i] for i in range(edge_index.shape[-1])], axis=1)

        # For ragged tensor we can now also try:
        # out = tf.gather(nod, edge_index, batch_dims=1) # Works now
        # if self.concat_nodes:
        #   out = tf.keras.backend.concatenate([out[:, :, i] for i in range(edge_index.shape[-1])], axis=2)

        return kgcnn_ops_dyn_cast([out, edge_part], input_tensor_type="values_partition",
                                  output_tensor_type=found_edge_type, partition_type=self.partition_type)

    def get_config(self):
        """Update config."""
        config = super(GatherNodes, self).get_config()
        config.update({"node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "concat_nodes": self.concat_nodes,
                       "input_tensor_type": self.input_tensor_type,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "ragged_validate": self.ragged_validate})
        return config


class GatherNodesOutgoing(ks.layers.Layer):
    """
    Gather nodes by edge edge_indices.
    
    If graphs edge_indices were in 'sample' mode, the edge_indices must be corrected for disjoint graphs.
    For outgoing nodes, layer uses only index[1].
    
    Args:
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes or edges to batch. Default is "row_length".
            This is used for input_tensor_type="values_partition".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
    """

    def __init__(self,
                 node_indexing='sample',
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 is_sorted=False,
                 has_unconnected=True,
                 **kwargs):
        """Initialize layer."""
        super(GatherNodesOutgoing, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]
        self._supports_ragged_inputs = True

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

    def build(self, input_shape):
        """Build layer."""
        super(GatherNodesOutgoing, self).build(input_shape)

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
            inputs (list): [nodes, edge_index]

            - nodes: Node embeddings of shape (batch, [N], F)
            - edge_index: Edge indices of shape (batch, [M], 2)

        Returns:
            embeddings: Gathered node embeddings that match the number of edges.
        """
        found_node_type = kgcnn_ops_check_tensor_type(inputs[0], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        found_edge_type = kgcnn_ops_check_tensor_type(inputs[1], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        # We cast to values here
        node, node_part = kgcnn_ops_dyn_cast(inputs[0], input_tensor_type=found_node_type,
                                             output_tensor_type="values_partition",
                                             partition_type=self.partition_type)
        edge_index, edge_part = kgcnn_ops_dyn_cast(inputs[1], input_tensor_type=found_edge_type,
                                                   output_tensor_type="values_partition",
                                                   partition_type=self.partition_type)

        indexlist = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edge_index, node_part, edge_part,
                                                                           partition_type_node=self.partition_type,
                                                                           partition_type_edge=self.partition_type,
                                                                           to_indexing='batch',
                                                                           from_indexing=self.node_indexing)
        out = tf.gather(node, indexlist[:, 1], axis=0)

        # For ragged tensor we can now also try:
        # out = tf.gather(nod, edge_index[:, :, 1], batch_dims=1)

        return kgcnn_ops_dyn_cast([out, edge_part], input_tensor_type="values_partition",
                                  output_tensor_type=found_edge_type, partition_type=self.partition_type)

    def get_config(self):
        """Update config."""
        config = super(GatherNodesOutgoing, self).get_config()
        config.update({"node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "ragged_validate": self.ragged_validate})
        return config


class GatherNodesIngoing(ks.layers.Layer):
    """
    Gather nodes by edge edge_indices.
    
    If graphs edge_indices were in 'sample' mode, the edge_indices must be corrected for disjoint graphs.
    For ingoing nodes, layer uses only index[0].
    
    Args:
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes or edges to batch. Default is "row_length".
            This is used for input_tensor_type="values_partition".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
    """

    def __init__(self,
                 node_indexing='sample',
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 is_sorted=False,
                 has_unconnected=True,
                 **kwargs):
        """Initialize layer."""
        super(GatherNodesIngoing, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]
        self._supports_ragged_inputs = True

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

    def build(self, input_shape):
        """Build layer."""
        super(GatherNodesIngoing, self).build(input_shape)

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
            inputs (list): [nodes, edge_index]

            - nodes: Node embeddings of shape (batch, [N], F)
            - edge_index: Edge indices of shape (batch, [M], 2)

        Returns:
            embeddings: Gathered node embeddings that match the number of edges.
        """
        found_node_type = kgcnn_ops_check_tensor_type(inputs[0], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        found_edge_type = kgcnn_ops_check_tensor_type(inputs[1], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)

        # We cast to values here
        node, node_part = kgcnn_ops_dyn_cast(inputs[0], input_tensor_type=found_node_type,
                                             output_tensor_type="values_partition",
                                             partition_type=self.partition_type)
        edge_index, edge_part = kgcnn_ops_dyn_cast(inputs[1], input_tensor_type=found_edge_type,
                                                   output_tensor_type="values_partition",
                                                   partition_type=self.partition_type)

        indexlist = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edge_index, node_part, edge_part,
                                                                           partition_type_node=self.partition_type,
                                                                           partition_type_edge=self.partition_type,
                                                                           to_indexing='batch',
                                                                           from_indexing=self.node_indexing)
        out = tf.gather(node, indexlist[:, 0], axis=0)

        # For ragged tensor we can now also try:
        # out = tf.gather(nod, edge_index[:, :, 0], batch_dims=1)

        return kgcnn_ops_dyn_cast([out, edge_part], input_tensor_type="values_partition",
                                  output_tensor_type=found_edge_type, partition_type=self.partition_type)

    def get_config(self):
        """Update config."""
        config = super(GatherNodesIngoing, self).get_config()
        config.update({"node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "ragged_validate": self.ragged_validate})
        return config


class GatherState(ks.layers.Layer):
    """
    Layer to repeat environment or global state for node or edge lists.
    
    To repeat the correct environment for each sample, a tensor with the target length/partition is required.

    Args:
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes or edges to batch. Default is "row_length".
            This is used for input_tensor_type="values_partition".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
    """

    def __init__(self,
                 node_indexing='sample',
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 is_sorted=False,
                 has_unconnected=True,
                 **kwargs):
        """Initialize layer."""
        super(GatherState, self).__init__(**kwargs)
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.ragged_validate = ragged_validate
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]
        self._supports_ragged_inputs = True

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

    def build(self, input_shape):
        """Build layer."""
        super(GatherState, self).build(input_shape)

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
            inputs: [state, target]

            - state: Graph specific embedding tensor. This is usually tensor of shape (batch, F.)
            - target: Target to collect state for. This can be node or edge embeddings of shape (batch, [N], F)

        Returns:
            state: Graph embedding with repeated single state for each graph.
        """
        found_state_type = kgcnn_ops_check_tensor_type(inputs[0], input_tensor_type="tensor",
                                                       node_indexing=self.node_indexing)
        found_ref_type = kgcnn_ops_check_tensor_type(inputs[1], input_tensor_type=self.input_tensor_type,
                                                     node_indexing=self.node_indexing)
        # We cast to values here
        env = kgcnn_ops_dyn_cast(inputs[0], input_tensor_type=found_state_type,
                                 output_tensor_type="tensor",
                                 partition_type=self.partition_type)
        _, target_part = kgcnn_ops_dyn_cast(inputs[1], input_tensor_type=found_ref_type,
                                            output_tensor_type="values_partition",
                                            partition_type=self.partition_type)

        target_len = kgcnn_ops_change_partition_type(target_part, self.partition_type, "row_length")

        out = tf.repeat(env, target_len, axis=0)

        return kgcnn_ops_dyn_cast([out, target_part], input_tensor_type="values_partition",
                                  output_tensor_type=found_ref_type, partition_type=self.partition_type)

    def get_config(self):
        """Update config."""
        config = super(GatherState, self).get_config()
        config.update({"node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "ragged_validate": self.ragged_validate})
        return config

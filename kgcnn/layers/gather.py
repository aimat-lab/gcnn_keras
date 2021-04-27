import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.ops.partition import kgcnn_ops_change_partition_type, kgcnn_ops_change_edge_tensor_indexing_by_row_partition


class GatherNodes(ks.layers.Layer):
    """
    Gather nodes by edge_indices. Index list must match node tensor.
    
    If graphs edge_indices were in 'sample' mode, the edge_indices must be corrected for disjoint graphs.
    
    Args:
        concat_nodes (bool): Whether to concatenate gathered node features. Default is True.
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes or edges to batch. Default is "row_length".
            This is used for input_tensor_type="values_partition".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        **kwargs
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
        super(GatherNodes, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [nodes, edge_index]

            - nodes: Node features.
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
            features: Gathered node features that match the shape of edge indices up to feature dimensions.
        """
        if self.input_tensor_type == "values_partition":
            [node, node_part], [edge_index, edge_part] = inputs
            indexlist = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edge_index, node_part, edge_part,
                                                                               partition_type_node=self.partition_type,
                                                                               partition_type_edge=self.partition_type,
                                                                               to_indexing='batch',
                                                                               from_indexing=self.node_indexing)
            out = tf.gather(node, indexlist, axis=0)
            if self.concat_nodes:
                out = tf.keras.backend.concatenate([out[:,i] for i in range(edge_index.shape[-1])],axis=1)
            return [out, edge_part]

        elif self.input_tensor_type == "ragged":
            nod, edge_index = inputs
            if self.node_indexing == 'batch':
                out = tf.RaggedTensor.from_row_splits(tf.gather(nod.values, edge_index.values), edge_index.row_splits,
                                                      validate=self.ragged_validate)
            elif self.node_indexing == 'sample':
                out = tf.gather(nod, edge_index, batch_dims=1)
            else:
                raise TypeError("Error: Unknown index convention, use: 'sample' or 'batch'.")
            if self.concat_nodes:
                out = tf.keras.backend.concatenate([out[:, :, i] for i in range(edge_index.shape[-1])], axis=2)
            return out

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
        **kwargs
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
        super(GatherNodesOutgoing, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [nodes, edge_index]

            - nodes: Node features.
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
            features: Gathered node features that match the shape of edge indices up to feature dimensions.
        """
        if self.input_tensor_type == "values_partition":
            [node, node_part], [edge_index, edge_part] = inputs
            # node,edge_index= inputs
            indexlist = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edge_index, node_part, edge_part,
                                                                               partition_type_node=self.partition_type,
                                                                               partition_type_edge=self.partition_type,
                                                                               to_indexing='batch',
                                                                               from_indexing=self.node_indexing)

            out = tf.gather(node, indexlist[:, 1], axis=0)
            return [out, edge_part]

        elif self.input_tensor_type == "ragged":
            nod, edge_index = inputs
            if self.node_indexing == 'batch':
                out = tf.RaggedTensor.from_row_splits(tf.gather(nod.values, edge_index.values[:, 1]),
                                                      edge_index.row_splits, validate=self.ragged_validate)
            elif self.node_indexing == 'sample':
                out = tf.gather(nod, edge_index[:, :, 1], batch_dims=1)
            else:
                raise TypeError("Error: Unknown index convention, use: 'sample' or 'batch'.")
            return out

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
        **kwargs
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
        super(GatherNodesIngoing, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [nodes, edge_index]

            - nodes: Node features.
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
            features: Gathered node features that match the shape of edge indices up to feature dimensions.
        """
        if self.input_tensor_type == "values_partition":
            [node, node_part], [edge_index, edge_part] = inputs
            # node,edge_index= inputs
            indexlist = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edge_index, node_part, edge_part,
                                                                               partition_type_node=self.partition_type,
                                                                               partition_type_edge=self.partition_type,
                                                                               to_indexing='batch',
                                                                               from_indexing=self.node_indexing)

            out = tf.gather(node, indexlist[:, 0], axis=0)
            return [out,edge_part]
        elif self.input_tensor_type == "ragged":
            nod, edge_index = inputs
            if self.node_indexing == 'batch':
                out = tf.RaggedTensor.from_row_splits(tf.gather(nod.values, edge_index.values[:, 0]),
                                                      edge_index.row_splits, validate=self.ragged_validate)
            elif self.node_indexing == 'sample':
                out = tf.gather(nod, edge_index[:, :, 0], batch_dims=1)
            else:
                raise TypeError("Error: Unknown index convention, use: 'sample' or 'batch'.")
            return out

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
        **kwargs
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
        super(GatherState, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [environment, target_length]

            - environment (tf.tensor): List of graph specific feature tensor of shape (batch*None,F)
            - target_partition (tf.tensor): Assignment of nodes or edges to each graph in batch.
              Default is row_length of shape (batch,).

        Returns:
            features (tf.tensor): A tensor with repeated single state for each graph.
            Output shape is (batch*N,F).
        """
        env, target_part = inputs

        target_len = kgcnn_ops_change_partition_type(target_part, self.partition_type, "row_length")

        out = tf.repeat(env, target_len, axis=0)
        return out

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

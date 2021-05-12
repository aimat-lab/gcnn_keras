import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.ops.casting import kgcnn_ops_dyn_cast
from kgcnn.ops.partition import kgcnn_ops_change_partition_type, kgcnn_ops_change_edge_tensor_indexing_by_row_partition
from kgcnn.ops.types import kgcnn_ops_static_test_tensor_input_type, kgcnn_ops_check_tensor_type


class CastRaggedToDisjointSparseAdjacency(tf.keras.layers.Layer):
    """
    Layer to cast e.g. RaggedTensor graph representation to a single Sparse tensor in disjoint representation.

    This includes edge_indices and adjacency matrix entries. The Sparse tensor is simply the adjacency matrix.

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
                 node_indexing="sample",
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 is_sorted=False,
                 has_unconnected=True,
                 **kwargs):
        """Initialize layer."""
        super(CastRaggedToDisjointSparseAdjacency, self).__init__(**kwargs)
        self.ragged_validate = ragged_validate
        self.is_sorted = is_sorted
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.has_unconnected = has_unconnected
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]
        self._supports_ragged_inputs = True

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

    def build(self, input_shape):
        """Build layer."""
        super(CastRaggedToDisjointSparseAdjacency, self).build(input_shape)

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
            Inputs list of [nodes, edges, edge_index]

            - nodes: Node feature tensor of shape (batch, [N], F)
            - edges: Edge feature ragged tensor of shape (batch, [N], 1)
            - edge_index: Ragged edge_indices of shape (batch, [N], 2)

        Returns:
            tf.sparse: Sparse disjoint matrix of shape (batch*None,batch*None)
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
        edge_index, edge_part = kgcnn_ops_dyn_cast(inputs[2], input_tensor_type=found_index_type,
                                                   output_tensor_type="values_partition",
                                                   partition_type=self.partition_type)

        # Cast to length tensor
        node_len = kgcnn_ops_change_partition_type(node_part, self.partition_type, "row_length")
        edge_len = kgcnn_ops_change_partition_type(edge_part, self.partition_type, "row_length")

        # batch-wise indexing
        edge_index = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edge_index,
                                                                            node_len, edge_len,
                                                                            partition_type_node="row_length",
                                                                            partition_type_edge="row_length",
                                                                            from_indexing=self.node_indexing,
                                                                            to_indexing="batch")
        indexlist = edge_index
        valuelist = edge

        if not self.is_sorted:
            # Sort per outgoing
            batch_order = tf.argsort(indexlist[:, 1], axis=0, direction='ASCENDING')
            indexlist = tf.gather(indexlist, batch_order, axis=0)
            valuelist = tf.gather(valuelist, batch_order, axis=0)
            # Sort per ingoing node
            node_order = tf.argsort(indexlist[:, 0], axis=0, direction='ASCENDING', stable=True)
            indexlist = tf.gather(indexlist, node_order, axis=0)
            valuelist = tf.gather(valuelist, node_order, axis=0)

        indexlist = tf.cast(indexlist, dtype=tf.int64)
        dense_shape = tf.concat([tf.shape(nod)[0:1], tf.shape(nod)[0:1]], axis=0)
        dense_shape = tf.cast(dense_shape, dtype=tf.int64)
        out = tf.sparse.SparseTensor(indexlist, valuelist[:, 0], dense_shape)

        return out

    def get_config(self):
        """Update layer config."""
        config = super(CastRaggedToDisjointSparseAdjacency, self).get_config()
        config.update({"node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "ragged_validate": self.ragged_validate})
        return config


class PoolingAdjacencyMatmul(ks.layers.Layer):
    r"""
    Layer for pooling of node features by multiplying with sparse adjacency matrix. Which gives $A n$.
    The node features needs to be flatten for a disjoint representation.

    Args:
        node_indexing (str): Indices referring to 'sample' or to the continuous 'batch'.
            For disjoint representation 'batch' is default.
        partition_type (str): Partition tensor type to assign nodes or edges to batch. Default is "row_length".
            This is used for input_tensor_type="values_partition".
        input_tensor_type (str): Input type of the tensors for call(). Default is "ragged".
        ragged_validate (bool): Whether to validate ragged tensor. Default is False.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        pooling_method (str): Not used. Default is "sum".
    """

    def __init__(self,
                 pooling_method="sum",
                 node_indexing="sample",
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 is_sorted=False,
                 has_unconnected=True,
                 **kwargs):
        """Initialize layer."""
        super(PoolingAdjacencyMatmul, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.ragged_validate = ragged_validate
        self.is_sorted = is_sorted
        self.node_indexing = node_indexing
        self.partition_type = partition_type
        self.input_tensor_type = input_tensor_type
        self.has_unconnected = has_unconnected
        self._tensor_input_type_implemented = ["ragged", "values_partition", "disjoint", "tensor", "RaggedTensor"]
        self._supports_ragged_inputs = True

        self._test_tensor_input = kgcnn_ops_static_test_tensor_input_type(self.input_tensor_type,
                                                                          self._tensor_input_type_implemented,
                                                                          self.node_indexing)

    def build(self, input_shape):
        """Build layer."""
        super(PoolingAdjacencyMatmul, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, adjacency]

            - nodes: Node features of shape (batch, [N], F)
            - adjacency (tf.sparse): SparseTensor of the adjacency matrix of shape (batch*None,batch*None)

        Returns:
            features (tf.tensor): Pooled node features of shape (batch,F)
        """
        adj = inputs[1]
        found_node_type = kgcnn_ops_check_tensor_type(inputs[0], input_tensor_type=self.input_tensor_type,
                                                      node_indexing=self.node_indexing)
        node, node_part = kgcnn_ops_dyn_cast(inputs[0], input_tensor_type=found_node_type,
                                             output_tensor_type="values_partition",
                                             partition_type=self.partition_type)

        out = tf.sparse.sparse_dense_matmul(adj, node)

        return kgcnn_ops_dyn_cast([out, node_part], input_tensor_type="values_partition",
                                  output_tensor_type=found_node_type, partition_type=self.partition_type)

    def get_config(self):
        """Update layer config."""
        config = super(PoolingAdjacencyMatmul, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "ragged_validate": self.ragged_validate})
        return config

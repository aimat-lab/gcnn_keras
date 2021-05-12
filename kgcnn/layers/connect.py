import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.ops.casting import kgcnn_ops_dyn_cast
from kgcnn.ops.partition import kgcnn_ops_change_partition_type, kgcnn_ops_change_edge_tensor_indexing_by_row_partition
from kgcnn.ops.types import kgcnn_ops_static_test_tensor_input_type, kgcnn_ops_check_tensor_type


# import tensorflow.keras.backend as ksb

class AdjacencyPower(ks.layers.Layer):
    """
    Computes powers of the adjacency matrix. This implementation is a temporary solution.
    
    Note: Layer casts to dense until sparse matmul is supported. This is very inefficient.
        
    Args:
        n (int): Power of the adjacency matrix. Default is 2.
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
                 n=2,
                 node_indexing="sample",
                 partition_type="row_length",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 is_sorted=False,
                 has_unconnected=True,
                 **kwargs):
        """Initialize layer."""
        super(AdjacencyPower, self).__init__(**kwargs)
        self.n = n
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
        super(AdjacencyPower, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward path.

        The tensor representation can be tf.RaggedTensor, tf.Tensor or a list of (values, partition).
        The RaggedTensor has shape (batch, None, F) or in case of equal sized graphs (batch, N, F).
        For disjoint representation (values, partition), the node embeddings are given by
        a flatten value tensor of shape (batch*None, F) and a partition tensor of either "row_length",
        "row_splits" or "value_rowids" that matches the tf.RaggedTensor partition information. In this case
        the partition_type and node_indexing scheme, i.e. "batch", must be known by the layer.
        For edge indices, the last dimension holds indices from outgoing to ingoing node (i,j) as a directed edge.

        Args:
            inputs (list): [nodes, edges, edge_indices]

            - nodes: Node emebeddings of shape (batch, [N], F)
            - edges: Adjacency entries of shape (batch, [N], 1)
            - edge_indices: Index list of shape (batch, [N], 2)
            
        Returns:
            list: [edges, edge_indices]

            - edges: Adjacency entries of shape  (batch, [N], 1)
            - edge_indices: Flatten index list of shape (batch, [N], 2)
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
                                                                            to_indexing="sample")

        ind_batch = tf.cast(tf.expand_dims(tf.repeat(tf.range(tf.shape(edge_len)[0]), edge_len), axis=-1),
                            dtype=edge_index.dtype)
        ind_all = tf.concat([ind_batch, edge_index], axis=-1)
        ind_all = tf.cast(ind_all, dtype=tf.int64)

        max_index = tf.reduce_max(node_len)
        dense_shape = tf.stack([tf.cast(tf.shape(node_len)[0], dtype=max_index.dtype), max_index, max_index])
        adj = tf.zeros(dense_shape, dtype=edge.dtype)
        ind_flat = tf.range(tf.cast(tf.shape(node_len)[0], dtype=max_index.dtype) * max_index * max_index)

        adj = tf.expand_dims(adj, axis=-1)
        adj = tf.tensor_scatter_nd_update(adj, ind_all, edge[:, 0:1])
        adj = tf.squeeze(adj, axis=-1)

        out0 = adj
        out = adj
        for i in range(self.n - 1):
            out = tf.linalg.matmul(out, out0)

        # debug_result = out

        # sparsify
        mask = out > tf.keras.backend.epsilon()
        mask = tf.reshape(mask, (-1,))
        out = tf.reshape(out, (-1,))

        new_edge = out[mask]
        new_edge = tf.expand_dims(new_edge, axis=-1)
        new_indices = tf.unravel_index(ind_flat[mask], dims=dense_shape)
        new_egde_ids = new_indices[0]
        new_edge_index = tf.concat([tf.expand_dims(new_indices[1], axis=-1), tf.expand_dims(new_indices[2], axis=-1)],
                                   axis=-1)
        new_edge_len = tf.tensor_scatter_nd_add(tf.zeros_like(node_len), tf.expand_dims(new_egde_ids, axis=-1),
                                                tf.ones_like(new_egde_ids))

        # Outpartition
        new_edge_part = kgcnn_ops_change_partition_type(new_edge_len, "row_length", self.partition_type)

        # batchwise indexing
        new_edge_index = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(new_edge_index,
                                                                                node_len, new_edge_len,
                                                                                partition_type_node="row_length",
                                                                                partition_type_edge="row_length",
                                                                                from_indexing="sample",
                                                                                to_indexing=self.node_indexing)

        outlist = [kgcnn_ops_dyn_cast([new_edge, new_edge_part], input_tensor_type="values_partition",
                                      output_tensor_type=found_edge_type, partition_type=self.partition_type),
                   kgcnn_ops_dyn_cast([new_edge_index, new_edge_part], input_tensor_type="values_partition",
                                      output_tensor_type=found_index_type, partition_type=self.partition_type)
                   ]

        return outlist

    def get_config(self):
        """Update layer config."""
        config = super(AdjacencyPower, self).get_config()
        config.update({"n": self.n,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type,
                       "input_tensor_type": self.input_tensor_type,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "ragged_validate": self.ragged_validate})
        return config

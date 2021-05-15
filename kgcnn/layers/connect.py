import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.partition import kgcnn_ops_change_edge_tensor_indexing_by_row_partition


# import tensorflow.keras.backend as ksb

class AdjacencyPower(GraphBaseLayer):
    """
    Computes powers of the adjacency matrix. This implementation is a temporary solution.
    
    Note: Layer casts to dense until sparse matmul is supported. This is very inefficient.
        
    Args:
        n (int): Power of the adjacency matrix. Default is 2.
    """

    def __init__(self, n=2, **kwargs):
        """Initialize layer."""
        super(AdjacencyPower, self).__init__(**kwargs)
        self.n = n

    def build(self, input_shape):
        """Build layer."""
        super(AdjacencyPower, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward path.

        Args:
            inputs (list): [nodes, edges, edge_indices]

            - nodes (tf.ragged): Node emebeddings of shape (batch, [N], F)
            - edges (tf.ragged): Adjacency entries of shape (batch, [M], 1)
            - edge_indices (tf.ragged): Index list of shape (batch, [M], 2)
            
        Returns:
            list: [edges, edge_indices]

            - edges (tf.ragged): Adjacency entries of shape  (batch, [M], 1)
            - edge_indices (tf.ragged): Flatten index list of shape (batch, [M], 2)
        """
        dyn_inputs = self._kgcnn_map_input_ragged(inputs, 3)

        nod, node_len = dyn_inputs[0].values, dyn_inputs[0].row_lengths()
        edge = dyn_inputs[1].values
        edge_index, edge_len = dyn_inputs[2].values, dyn_inputs[2].row_lengths()

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

        # batchwise indexing
        new_edge_index = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(new_edge_index,
                                                                                node_len, new_edge_len,
                                                                                partition_type_node="row_length",
                                                                                partition_type_edge="row_length",
                                                                                from_indexing="sample",
                                                                                to_indexing=self.node_indexing)

        outlist = [self._kgcnn_map_output_ragged([new_edge, new_edge_len], "row_length", 1),
                   self._kgcnn_map_output_ragged([new_edge_index, new_edge_len], "row_length", 2)]

        return outlist

    def get_config(self):
        """Update layer config."""
        config = super(AdjacencyPower, self).get_config()
        config.update({"n": self.n})
        return config

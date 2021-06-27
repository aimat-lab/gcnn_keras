import tensorflow as tf
# import tensorflow.keras.backend as ksb
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.partition import partition_row_indexing


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='AdjacencyPower')
class AdjacencyPower(GraphBaseLayer):
    """Computes powers of the adjacency matrix. This implementation is a temporary solution.
    
    Note: Layer casts to dense until sparse matmul is supported. This can be very inefficient.
        
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
        """Forward pass.

        Args:
            inputs (list): [nodes, edges, edge_indices]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Adjacency entries of shape (batch, [M], 1)
                - edge_indices (tf.RaggedTensor): Edge-index list referring to nodes of shape (batch, [M], 2)
            
        Returns:
            list: [edges, edge_indices]

                - edges (tf.RaggedTensor): Adjacency entries of shape  (batch, [M], 1)
                - edge_indices (tf.RaggedTensor): Flatten index list of shape (batch, [M], 2)
        """
        dyn_inputs = inputs

        nod, node_len = dyn_inputs[0].values, dyn_inputs[0].row_lengths()
        edge = dyn_inputs[1].values
        edge_index, edge_len = dyn_inputs[2].values, dyn_inputs[2].row_lengths()

        # batch-wise indexing
        edge_index = partition_row_indexing(edge_index,
                                            node_len, edge_len,
                                            partition_type_target="row_length",
                                            partition_type_index="row_length",
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

        # Make sparse
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
        new_edge_index = partition_row_indexing(new_edge_index,
                                                node_len, new_edge_len,
                                                partition_type_target="row_length",
                                                partition_type_index="row_length",
                                                from_indexing="sample",
                                                to_indexing=self.node_indexing)

        outlist = [tf.RaggedTensor.from_row_lengths(new_edge, new_edge_len, validate=self.ragged_validate),
                   tf.RaggedTensor.from_row_lengths(new_edge_index, new_edge_len, validate=self.ragged_validate)]

        return outlist

    def get_config(self):
        """Update layer config."""
        config = super(AdjacencyPower, self).get_config()
        config.update({"n": self.n})
        return config

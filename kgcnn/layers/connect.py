import tensorflow as tf
import tensorflow.keras as ks
# import tensorflow.keras.backend as ksb

from kgcnn.ops.partition import kgcnn_ops_change_partition_type, kgcnn_ops_change_edge_tensor_indexing_by_row_partition
from kgcnn.ops.casting import kgcnn_ops_cast_ragged_to_value_partition

class AdjacencyPower(ks.layers.Layer):
    """
    Computes powers of the adjacency matrix. This implementation is a temporary solution.
    
    Note: Layer casts to dense until sparse matmul is supported. This is very inefficient.
        
    Args:
        n (int): Power of the adjacency matrix. Default is 2.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,
                 n=2,
                 partition_type="row_length",
                 node_indexing="sample",
                 input_tensor_type="ragged",
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(AdjacencyPower, self).__init__(**kwargs)
        self.n = n
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
        super(AdjacencyPower, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward path.

        Args:
            inputs (list): [edge_indices, edges, edge_length, node_length]

            - nodes: Node emebeddings
            - edges: adjacency entries of shape (batch*None,1)
            - edge_indices: Flatten index list of shape (batch*None,2)
            
        Returns:
            list: [edges, edge_indices]

            - edges (tf.tensor): Flatten adjacency entries of shape (batch*None,1)
            - edge_indices (tf.tensor): Flatten index list of shape (batch*None,2)
        """
        if self.input_tensor_type == "values_partition":
            [nod, node_part], [edge, _], [edge_index, edge_part] = inputs
        elif self.input_tensor_type == "ragged":
            nod, node_part = kgcnn_ops_cast_ragged_to_value_partition(inputs[0], self.partition_type)
            edge, _ = kgcnn_ops_cast_ragged_to_value_partition(inputs[1], self.partition_type)
            edge_index, edge_part = kgcnn_ops_cast_ragged_to_value_partition(inputs[2], self.partition_type)
        else:
            raise NotImplementedError("Error: Not supported tensor input.")


        # Cast to length tensor
        node_len = kgcnn_ops_change_partition_type(node_part, self.partition_type, "row_length")
        edge_len = kgcnn_ops_change_partition_type(edge_part, self.partition_type, "row_length")

        # batchwise indexing
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

        max_index = tf.reduce_max(edge_len)
        dense_shape = tf.stack([tf.cast(tf.shape(edge_len)[0], dtype=max_index.dtype), max_index, max_index])
        dense_shape = tf.cast(dense_shape, dtype=tf.int64)

        edge_val = edge[:, 0]  # Must be 1D tensor
        adj = tf.sparse.SparseTensor(ind_all, edge_val, dense_shape)

        out0 = tf.sparse.to_dense(adj, validate_indices=False)
        out = out0

        for i in range(self.n - 1):
            out = tf.matmul(out, out0)

        ind1 = tf.repeat(tf.expand_dims(tf.range(max_index), axis=-1), max_index, axis=-1)
        ind2 = tf.repeat(tf.expand_dims(tf.range(max_index), axis=0), max_index, axis=0)
        ind12 = tf.concat([tf.expand_dims(ind1, axis=-1), tf.expand_dims(ind2, axis=-1)], axis=-1)
        ind = tf.repeat(tf.expand_dims(ind12, axis=0), tf.shape(edge_len)[0], axis=0)
        new_shift = tf.expand_dims(
            tf.expand_dims(tf.expand_dims(tf.cumsum(node_len, exclusive=True), axis=-1), axis=-1), axis=-1)
        ind = ind + new_shift

        mask = out > 0
        imask = tf.cast(mask, dtype=max_index.dtype)
        new_edge_len = tf.reduce_sum(tf.reduce_sum(imask, axis=-1), axis=-1)

        new_edge_index = ind[mask]
        new_edge = tf.expand_dims(out[mask], axis=-1)

        # Outpartition
        new_edge_part = kgcnn_ops_change_partition_type(new_edge_len, "row_length", self.partition_type)

        # batchwise indexing
        edge_index = kgcnn_ops_change_edge_tensor_indexing_by_row_partition(new_edge_index,
                                                                            node_len, new_edge_len,
                                                                            partition_type_node="row_length",
                                                                            partition_type_edge="row_length",
                                                                            from_indexing="batch",
                                                                            to_indexing=self.node_indexing)

        if self.input_tensor_type == "values_partition":
            return [new_edge, new_edge_part], [new_edge_index, new_edge_part]
        elif self.input_tensor_type == "ragged":
            outlist = [tf.RaggedTensor.from_row_lengths(new_edge, new_edge_len, validate=self.ragged_validate),
                       tf.RaggedTensor.from_row_lengths(new_edge_index, new_edge_len, validate=self.ragged_validate)]
            return outlist

    def get_config(self):
        """Update layer config."""
        config = super(AdjacencyPower, self).get_config()
        config.update({"n": self.n,
                       "partition_type": self.partition_type,
                       "node_indexing": self.node_indexing,
                       "input_tensor_type" : self.input_tensor_type,
                       "ragged_validate" : self.ragged_validate
                       })
        return config

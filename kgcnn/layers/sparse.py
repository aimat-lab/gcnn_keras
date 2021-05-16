import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.partition import kgcnn_ops_change_edge_tensor_indexing_by_row_partition


class CastRaggedToDisjointSparseAdjacency(GraphBaseLayer):
    """Layer to cast e.g. RaggedTensor graph representation to a single Sparse tensor in disjoint representation.

    This includes edge_indices and adjacency matrix entries. The Sparse tensor is simply the adjacency matrix.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(CastRaggedToDisjointSparseAdjacency, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(CastRaggedToDisjointSparseAdjacency, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            Inputs list of [nodes, edges, edge_index]

            - nodes (tf.ragged): Node feature tensor of shape (batch, [N], F)
            - edges (tf.ragged): Edge feature ragged tensor of shape (batch, [M], 1)
            - edge_index (tf.ragged): Ragged edge_indices of shape (batch, [M], 2)

        Returns:
            tf.sparse: Sparse disjoint matrix of shape (batch*None,batch*None)
        """
        dyn_inputs = self._kgcnn_map_input_ragged(inputs, 3)
        # We cast to values here
        nod, node_len = dyn_inputs[0].values, dyn_inputs[0].row_lengths()
        edge, _ = dyn_inputs[1].values, dyn_inputs[1].row_lengths()
        edge_index, edge_len = dyn_inputs[2].values, dyn_inputs[2].row_lengths()

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
        return config


class PoolingAdjacencyMatmul(GraphBaseLayer):
    r"""Layer for pooling of node features by multiplying with sparse adjacency matrix. Which gives $A n$.

    The node features needs to be flatten for a disjoint representation.

    Args:
        pooling_method (str): Not used. Default is "sum".
    """

    def __init__(self, pooling_method="sum", **kwargs):
        """Initialize layer."""
        super(PoolingAdjacencyMatmul, self).__init__(**kwargs)
        self.pooling_method = pooling_method

    def build(self, input_shape):
        """Build layer."""
        super(PoolingAdjacencyMatmul, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, adjacency]

            - nodes (tf.ragged): Node features of shape (batch, [N], F)
            - adjacency (tf.sparse): SparseTensor of the adjacency matrix of shape (batch*None, batch*None)

        Returns:
            features (tf.ragged): Pooled node features of shape (batch, [N], F)
        """
        adj = inputs[1]
        dyn_inputs = self._kgcnn_map_input_ragged([inputs[0]], 0)
        node, node_part = dyn_inputs[0].values, dyn_inputs[0].row_splits

        out = tf.sparse.sparse_dense_matmul(adj, node)
        out = self._kgcnn_map_output_ragged([out, node_part], "row_splits", 0)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingAdjacencyMatmul, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config

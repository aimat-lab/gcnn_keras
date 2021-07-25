import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.ops.partition import partition_row_indexing


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
            inputs (list): [nodes, edges, edge_index]

                - nodes (tf.RaggedTensor): Node feature tensor of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge feature ragged tensor of shape (batch, [M], 1)
                - edge_index (tf.RaggedTensor): Ragged edge_indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.SparseTensor: Sparse disjoint matrix of shape (batch*None,batch*None)
        """
        dyn_inputs = inputs
        # We cast to values here
        nod, node_len = dyn_inputs[0].values, dyn_inputs[0].row_lengths()
        edge, _ = dyn_inputs[1].values, dyn_inputs[1].row_lengths()
        edge_index, edge_len = dyn_inputs[2].values, dyn_inputs[2].row_lengths()

        # batch-wise indexing
        edge_index = partition_row_indexing(edge_index,
                                            node_len, edge_len,
                                            partition_type_target="row_length",
                                            partition_type_index="row_length",
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
    r"""Layer for pooling of node features by multiplying with sparse adjacency matrix.

    Computes

    The node features are flatten for a disjoint representation.

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
            inputs (list): [nodes, adjacency]

                - nodes (tf.RaggedTensor): Node features of shape (batch, [N], F)
                - adjacency (tf.SparseTensor): SparseTensor of the adjacency matrix of shape (batch*None, batch*None)

        Returns:
            tf.RaggedTensor: Pooled node features of shape (batch, [N], F)
        """
        adj = inputs[1]
        dyn_inputs = self._kgcnn_inspect_input_ragged([inputs[0]], 0)
        node, node_part = dyn_inputs[0].values, dyn_inputs[0].row_splits

        out = tf.sparse.sparse_dense_matmul(adj, node)
        out = tf.RaggedTensor.from_row_splits(out, node_part, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingAdjacencyMatmul, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config

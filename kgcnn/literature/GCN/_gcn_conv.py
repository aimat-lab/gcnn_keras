import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer


class GCNConvPoolingSparseAdjacencyMatmul(GraphBaseLayer):
    r"""Layer for graph convolution of node embeddings by multiplying with sparse adjacency matrix, as proposed
    in Graph convolution according to `Kipf et al <https://arxiv.org/abs/1609.02907>`_ .

    :math:`A x`, where :math:`A` represents the possibly scaled adjacency matrix.

    The node features are flatten for a disjoint representation.

    Args:
        pooling_method (str): Not used. Default is "sum".
    """

    def __init__(self, pooling_method="sum", **kwargs):
        """Initialize layer."""
        super(GCNConvPoolingSparseAdjacencyMatmul, self).__init__(**kwargs)
        self.pooling_method = pooling_method

    def build(self, input_shape):
        """Build layer."""
        super(GCNConvPoolingSparseAdjacencyMatmul, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [nodes, adjacency]

                - nodes (tf.RaggedTensor): Node features of shape (batch, [N], F)
                - adjacency (tf.SparseTensor): SparseTensor of the adjacency matrix of shape (batch*[N], batch*[N])

        Returns:
            tf.RaggedTensor: Pooled node features of shape (batch, [N], F)
        """
        inputs = self.assert_ragged_input_rank(inputs[0])
        adj = inputs[1]
        node, node_part = inputs[0].values, inputs[0].row_splits
        out = tf.sparse.sparse_dense_matmul(adj, node)
        out = tf.RaggedTensor.from_row_splits(out, node_part, validate=self.ragged_validate)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(GCNConvPoolingSparseAdjacencyMatmul, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config
import tensorflow as tf
import tensorflow.keras as ks


class PoolingAdjacencyMatmul(ks.layers.Layer):
    r"""
    Layer for pooling of node features by multiplying with sparse adjacency matrix. Which gives $A n$.
    The node features needs to be flatten for a disjoint representation.

    Args:
        pooling_method : tf.function to pool all nodes compatible with ragged tensors.
        **kwargs
    """

    def __init__(self,
                 pooling_method="sum",
                 **kwargs):
        """Initialize layer."""
        super(PoolingAdjacencyMatmul, self).__init__(**kwargs)
        self.pooling_method = pooling_method

    def build(self, input_shape):
        """Build layer."""
        super(PoolingAdjacencyMatmul, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            list: [adjacency, nodes]

            - nodes (tf.tensor): Flatten node features of shape (batch*None,F)
            - adjacency (tf.sparse): SparseTensor of the adjacency matrix of shape (batch*None,batch*None)

        Returns:
            features (tf.tensor): Pooled node features of shape (batch,F)
        """
        node, adj = inputs
        out = tf.sparse.sparse_dense_matmul(adj, node)

        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingAdjacencyMatmul, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config
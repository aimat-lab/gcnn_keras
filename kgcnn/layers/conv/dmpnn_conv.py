import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing
from kgcnn.layers.pool.pooling import PoolingLocalEdges


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='DMPNNPPoolingEdgesDirected')
class DMPNNPPoolingEdgesDirected(GraphBaseLayer):
    """Make a zero-like graph tensor. Calls tf.zeros_like()."""

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(DMPNNPPoolingEdgesDirected, self).__init__(**kwargs)
        self.pool_edge_1 = PoolingLocalEdges(pooling_method="sum")
        self.gather_edges = GatherNodesOutgoing()
        self.gather_pairs = GatherNodesIngoing()

    def build(self, input_shape):
        """Build layer."""
        super(DMPNNPPoolingEdgesDirected, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, edge_index, edge_reverse_pair]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)
                - edge_reverse_pair (tf.RaggedTensor): Pair mappings for reverse edges (batch, [M], 1)

        Returns:
            tf.RaggedTensor: Edge embeddings of shape (batch, [M], F)
        """
        n, ed, edi, edp = inputs
        pool_edge_receive = self.pool_edge_1([n, ed, edi])  # Sum pooling of all edges
        ed_new = self.gather_edges([pool_edge_receive, edi])
        ed_not = self.gather_pairs([ed, edp])
        out = ed_new - ed_not
        return out

import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing, GatherEdgesPairs
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.modules import LazySubtract


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='DMPNNPPoolingEdgesDirected')
class DMPNNPPoolingEdgesDirected(GraphBaseLayer):
    """Pooling of edges for around a target node as defined by
    `DMPNN <https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237>`__ . This slightly different than the normal node
    aggregation from message passing like networks. Requires edge pairs for this implementation.

    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(DMPNNPPoolingEdgesDirected, self).__init__(**kwargs)
        self.pool_edge_1 = AggregateLocalEdges(pooling_method="sum")
        self.gather_edges = GatherNodesOutgoing()
        self.gather_pairs = GatherEdgesPairs()
        self.subtract_layer = LazySubtract()

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
        pool_edge_receive = self.pool_edge_1([n, ed, edi], **kwargs)  # Sum pooling of all edges
        ed_new = self.gather_edges([pool_edge_receive, edi], **kwargs)
        ed_not = self.gather_pairs([ed, edp],  **kwargs)
        out = self.subtract_layer([ed_new, ed_not],  **kwargs)
        return out

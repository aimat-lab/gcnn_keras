import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.modules import LazySubtract


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='DMPNNGatherEdgesPairs')
class DMPNNGatherEdgesPairs(GraphBaseLayer):
    """Gather edge pairs that also works for invalid indices given a certain pair, i.e. if a edge does not have its
    reverse counterpart in the edge indices list.
    This class is used in `DMPNN <https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237>`_ .

    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(DMPNNGatherEdgesPairs, self).__init__(**kwargs)
        self.gather_layer = GatherNodesIngoing()

    def build(self, input_shape):
        """Build layer."""
        super(DMPNNGatherEdgesPairs, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): [edges, pair_index]

                - edges (tf.RaggedTensor): Node embeddings of shape (batch, [M], F)
                - pair_index (tf.RaggedTensor): Edge indices referring to edges of shape (batch, [M], 1)

        Returns:
            list: Gathered edge embeddings that match the reverse edges of shape (batch, [M], F) for selection_index.
        """
        self.assert_ragged_input_rank(inputs)
        edges, pair_index = inputs
        index_corrected = tf.RaggedTensor.from_row_splits(
            tf.where(pair_index.values >= 0, pair_index.values, tf.zeros_like(pair_index.values)),
            pair_index.row_splits, validate=self.ragged_validate)
        edges_paired = self.gather_layer([edges, index_corrected], **kwargs)
        edges_corrected = tf.RaggedTensor.from_row_splits(
            tf.where(pair_index.values >= 0, edges_paired.values, tf.zeros_like(edges_paired.values)),
            edges_paired.row_splits, validate=self.ragged_validate)
        return edges_corrected


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='DMPNNPPoolingEdgesDirected')
class DMPNNPPoolingEdgesDirected(GraphBaseLayer):
    """Pooling of edges for around a target node as defined by
    `DMPNN <https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237>`_ . This slightly different than the normal node
    aggregation from message passing like networks. Requires edge pairs for this implementation.

    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(DMPNNPPoolingEdgesDirected, self).__init__(**kwargs)
        self.pool_edge_1 = PoolingLocalEdges(pooling_method="sum")
        self.gather_edges = GatherNodesOutgoing()
        self.gather_pairs = DMPNNGatherEdgesPairs()
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

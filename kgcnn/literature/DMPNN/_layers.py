import keras as ks
from kgcnn.layers.gather import GatherNodesOutgoing, GatherEdgesPairs
from kgcnn.layers.aggr import AggregateLocalEdges
from keras.layers import Subtract


class DMPNNPPoolingEdgesDirected(ks.layers.Layer):  # noqa
    """Pooling of edges for around a target node as defined by

    `DMPNN <https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237>`__ . This is slightly different as the normal node
    aggregation from message passing like networks. Requires edge pair indices for this implementation.
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(DMPNNPPoolingEdgesDirected, self).__init__(**kwargs)
        self.pool_edge_1 = AggregateLocalEdges(pooling_method="scatter_sum")
        self.gather_edges = GatherNodesOutgoing()
        self.gather_pairs = GatherEdgesPairs()
        self.subtract_layer = Subtract()

    def build(self, input_shape):
        super(DMPNNPPoolingEdgesDirected, self).build(input_shape)
        # Could call build on sub-layers but is not necessary.

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edges, edge_index, edge_reverse_pair]

                - nodes (Tensor): Node embeddings of shape ([N], F)
                - edges (Tensor): Edge or message embeddings of shape ([M], F)
                - edge_index (Tensor): Edge indices referring to nodes of shape (2, [M])
                - edge_reverse_pair (Tensor): Pair mappings for reverse edges (1, [M])

        Returns:
            Tensor: Edge embeddings of shape ([M], F)
        """
        n, ed, edi, edp = inputs
        pool_edge_receive = self.pool_edge_1([n, ed, edi], **kwargs)  # Sum pooling of all edges
        ed_new = self.gather_edges([pool_edge_receive, edi], **kwargs)
        ed_not = self.gather_pairs([ed, edp],  **kwargs)
        out = self.subtract_layer([ed_new, ed_not],  **kwargs)
        return out

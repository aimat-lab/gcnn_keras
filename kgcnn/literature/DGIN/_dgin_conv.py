import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing, GatherEdgesPairs
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.modules import LazySubtract,LazyAdd


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

@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GIN_D')
class GIN_D(GraphBaseLayer):
    r"""Convolutional unit of `Graph Isomorphism Network from: How Powerful are Graph Neural Networks?
    <https://arxiv.org/abs/1810.00826>`_.
    modified to use h_w_0
    Computes graph convolution at step :math:`k` for node embeddings :math:`h_\nu` as:

    .. math::
        h_\nu^{(k)} = \phi^{(k)} ((1+\epsilon^{(k)}) h_\nu^{0} + \sum_{u\in N(\nu)}) h_u^{k-1}.

    with optional learnable :math:`\epsilon^{(k)}`

    .. note::
        The non-linear mapping :math:`\phi^{(k)}`, usually an :obj:`MLP`, is not included in this layer.
    """

    def __init__(self,
                 pooling_method='sum',
                 epsilon_learnable=False,
                 **kwargs):
        """Initialize layer.

        Args:
            epsilon_learnable (bool): If epsilon is learnable or just constant zero. Default is False.
            pooling_method (str): Pooling method for summing edges. Default is 'segment_sum'.
        """
        super(GIN_D, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.epsilon_learnable = epsilon_learnable

        # Layers
        self.lay_gather = GatherNodesOutgoing()
        self.lay_pool = AggregateLocalEdges(pooling_method=self.pooling_method)
        self.lay_add = LazyAdd()

        # Epsilon with trainable as optional and default zeros initialized.
        self.eps_k = self.add_weight(name="epsilon_k", trainable=self.epsilon_learnable,
                                     initializer="zeros", dtype=self.dtype)

    def build(self, input_shape):
        """Build layer."""
        super(GIN_D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.
        Args:
            inputs: [nodes, edge_index]
                - nodes (tf.RaggedTensor): Node embeddings of shape `(batch, [N], F)`
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape `(batch, [M], 2)`
                - nodes_0 (tf.RaggedTensor): Node embeddings of shape `(batch, [N], F)`

        Returns:
            tf.RaggedTensor: Node embeddings of shape `(batch, [N], F)`
        
        """
        node,  edge_index, node_0 = inputs # need to check if edge_index is full an not half (directed)
        ed = self.lay_gather([node, edge_index], **kwargs)
        nu = self.lay_pool([node, ed, edge_index], **kwargs)  # Summing for each node connection 
        no = (1+self.eps_k)*node_0 # modified to use node_0 instead of node see equation 7
        out = self.lay_add([no, nu], **kwargs)
        return out

    def get_config(self):
        """Update config."""
        config = super(GIN_D, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "epsilon_learnable": self.epsilon_learnable})
        return config


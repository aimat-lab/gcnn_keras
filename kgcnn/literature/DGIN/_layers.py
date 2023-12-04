from keras import ops
from kgcnn.layers.gather import GatherNodesOutgoing, GatherEdgesPairs
from kgcnn.layers.aggr import AggregateLocalEdges
from keras.layers import Subtract, Add, Layer


class DMPNNPPoolingEdgesDirected(Layer):  # noqa
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


class GIN_D(Layer):
    r"""Convolutional unit of `Graph Isomorphism Network from: How Powerful are Graph Neural Networks?
    <https://arxiv.org/abs/1810.00826>`__ .

    Modified to use :math:`h_{w_0}`

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
        self.lay_add = Add()

        # Epsilon with trainable as optional and default zeros initialized.
        self.eps_k = self.add_weight(name="epsilon_k", trainable=self.epsilon_learnable,
                                     initializer="zeros", dtype=self.dtype)

    def build(self, input_shape):
        """Build layer."""
        super(GIN_D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.
        Args:
            inputs: [nodes, edge_index, nodes_0]

                - nodes (Tensor): Node embeddings of shape `([N], F)`
                - edge_index (Tensor): Edge indices referring to nodes of shape `(2, [M])`
                - nodes_0 (Tensor): Node embeddings of shape `([N], F)`

        Returns:
            Tensor: Node embeddings of shape `([N], F)`
        """
        # Need to check if edge_index is full and not half (directed).
        node, edge_index, node_0 = inputs
        ed = self.lay_gather([node, edge_index], **kwargs)
        # Summing for each node connection
        nu = self.lay_pool([node, ed, edge_index], **kwargs)
        # Modified to use node_0 instead of node see equation 7 in paper.
        no = (ops.convert_to_tensor(1, dtype=self.eps_k.dtype) + self.eps_k) * node_0
        out = self.lay_add([no, nu], **kwargs)
        return out

    def get_config(self):
        """Update config."""
        config = super(GIN_D, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "epsilon_learnable": self.epsilon_learnable})
        return config

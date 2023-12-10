import keras as ks
from keras import ops
from keras.layers import Layer
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.aggr import AggregateLocalEdges
from keras.layers import Add, Concatenate


class rGIN(Layer):
    r"""Random Features Strengthen Graph Neural Networks <https://arxiv.org/abs/2002.03155>`__ .

    Computes graph convolution at step :math:`k` for node embeddings :math:`h_\nu` as:

    .. math::
        h_\nu^{(k)} = \phi^{(k)} ((1+\epsilon^{(k)}) h_\nu^{k-1} + \sum_{u\in N(\nu)}) h_u^{k-1}.

    with optional learnable :math:`\epsilon^{(k)}`

    .. note::

        The non-linear mapping :math:`\phi^{(k)}`, usually an :obj:`MLP`, is not included in this layer.
    """

    def __init__(self,
                 pooling_method='sum',
                 epsilon_learnable=False,
                 random_range=100,
                 **kwargs):
        """Initialize layer.

        Args:
            epsilon_learnable (bool): If epsilon is learnable or just constant zero. Default is False.
            pooling_method (str): Pooling method for summing edges. Default is 'segment_sum'.
        """
        super(rGIN, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.epsilon_learnable = epsilon_learnable
        self.random_range = random_range

        # Layers
        self.lay_gather = GatherNodesOutgoing()
        self.lay_pool = AggregateLocalEdges(pooling_method=self.pooling_method)
        self.lay_add = Add()
        self.lay_concat = Concatenate()

        # Epsilon with trainable as optional and default zeros initialized.
        self.eps_k = self.add_weight(
            name="epsilon_k",
            shape=(),
            trainable=self.epsilon_learnable,
            initializer="zeros",
            dtype=self.dtype
        )

    def build(self, input_shape):
        """Build layer."""
        super(rGIN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs: [nodes, edge_index]

                - nodes (Tensor): Node embeddings of shape `([N], F)`
                - edge_index (Tensor): Edge indices referring to nodes of shape `(2, [M])`

        Returns:
            Tensor: Node embeddings of shape `([N], F)`
        """
        node, edge_index = inputs
        node_shape = ops.shape(node)
        random_values = ops.cast(
            ks.random.uniform([node_shape[0], 1], maxval=self.random_range, dtype="float32"),
        self.dtype) / self.random_range

        node = self.lay_concat([node, random_values])

        ed = self.lay_gather([node, edge_index], **kwargs)
        nu = self.lay_pool([node, ed, edge_index], **kwargs)  # Summing for each node connection
        no = (1 + self.eps_k) * node

        # Add the random features to the original features
        out = self.lay_add([node, nu], **kwargs)

        return out

    def get_config(self):
        """Update config."""
        config = super(rGIN, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "epsilon_learnable": self.epsilon_learnable,
                       "random_range": self.random_range})
        return config

import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.pooling import PoolingLocalEdges
from kgcnn.layers.modules import LazyAdd


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='GIN')
class GIN(GraphBaseLayer):
    r"""Convolutional unit of `Graph Isomorphism Network from: How Powerful are Graph Neural Networks?
    <https://arxiv.org/abs/1810.00826>`_ .

    Computes graph convolution as:

    :math:`h_\nu^{(k)} = \phi^{(k)} ((1+\epsilon^{(k)}) h_\nu^{k-1} + \sum\limits_{u\in N(\nu)}) h_u^{k-1}`.
    with optional learnable :math:`\epsilon^{(k)}`
    Note: The non-linear mapping :math:`\phi^{(k)}`, usually an MLP, is not included in this layer.

    Args:
        epsilon_learnable (bool): If epsilon is learnable or just constant zero. Default is False.
        pooling_method (str): Pooling method for summing edges. Default is 'segment_sum'.
    """

    def __init__(self,
                 pooling_method='sum',
                 epsilon_learnable=False,
                 **kwargs):
        """Initialize layer."""
        super(GIN, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.epsilon_learnable = epsilon_learnable

        # Layers
        self.lay_gather = GatherNodesOutgoing()
        self.lay_pool = PoolingLocalEdges(pooling_method=self.pooling_method)
        self.lay_add = LazyAdd()

        # Epsilon with trainable as optional and default zeros initialized.
        self.eps_k = self.add_weight(name="epsilon_k", trainable=self.epsilon_learnable,
                                     initializer="zeros", dtype=self.dtype)

    def build(self, input_shape):
        """Build layer."""
        super(GIN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs: [nodes, edge_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Node embeddings of shape (batch, [N], F)
        """
        node, edge_index = inputs
        ed = self.lay_gather([node, edge_index], **kwargs)
        nu = self.lay_pool([node, ed, edge_index], **kwargs)  # Summing for each node connection
        no = (1+self.eps_k)*node
        # no = node
        out = self.lay_add([no, nu], **kwargs)
        return out

    def get_config(self):
        """Update config."""
        config = super(GIN, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "epsilon_learnable": self.epsilon_learnable})
        return config

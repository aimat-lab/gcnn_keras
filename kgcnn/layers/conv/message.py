import tensorflow as tf

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherEmbeddingSelection
from kgcnn.layers.pooling import PoolingLocalEdges


@tf.keras.utils.register_keras_serializable(package='kgcnn', name='MessagePassingBase')
class MessagePassingBase(GraphBaseLayer):
    r"""Base layer for Message passing type networks. This is a general frame to implement custom message and
    update functions. The idea is to create a subclass of :obj:`MessagePassingBase` and then just implement the methods
    :obj:`message_function` and :obj:`update_nodes`. The pooling is handled by built-in :obj:`PoolingLocalEdges`.
    Alternatively also :obj:`aggregate_message` could be overwritten.
    The original message passing scheme was proposed by `NMPNN <http://arxiv.org/abs/1704.01212>`_ .

    Args:
        pooling_method (str): Aggregation method of edges. Default is "sum".
    """

    def __init__(self, pooling_method="sum", **kwargs):
        """Initialize MessagePassingBase layer."""
        super(MessagePassingBase, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.lay_gather = GatherEmbeddingSelection([0, 1])
        self.lay_pool_default = PoolingLocalEdges(pooling_method=self.pooling_method)

    def message_function(self, inputs, **kwargs):
        r"""Defines the message function, i.e. a method the generates a message from node and edge embeddings at a
        certain depth (not considered here).

        Args:
            inputs: [nodes_in, nodes_out, edge_index]

                - nodes (tf.RaggedTensor): Receiving node embeddings of shape (batch, [N], F)
                - nodes_out (tf.RaggedTensor): Sending node embeddings of shape (batch, [N], F)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Messages for each edge of shape (batch, [M], F)
        """
        n_in, n_out, edges = inputs
        raise NotImplementedError(
            "A method to generate messages must be implemented in sub-class of `MessagePassingBase`.")

    def aggregate_message(self, inputs, **kwargs):
        r"""Pre-defined message aggregation that uses :obj:`PoolingLocalEdges`.

        Args:
            inputs: [nodes, edges, edge_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Aggregated edge embeddings per node of shape (batch, [N], F)
        """
        return self.lay_pool_default(inputs, **kwargs)

    def update_nodes(self, inputs, **kwargs):
        r"""Defines the update function, i.e. a method that updates the node embeddings from aggregated messages.

        Args:
            inputs: [nodes, node_updates]

                - nodes (tf.RaggedTensor): Node embeddings (from previous step) of shape (batch, [N], F)
                - node_updates (tf.RaggedTensor): Updates for nodes of shape (batch, [N], F)

        Returns:
            tf.RaggedTensor: Updated node embeddings (for next step) of shape (batch, [N], F)
        """
        nodes, nodes_update = inputs
        raise NotImplementedError(
            "A method to update nodes must be implemented in sub-class of `MessagePassingBase`.")

    def call(self, inputs, **kwargs):
        r"""Pre-implemented standard message passing scheme using :obj:`update_nodes`, :obj:`aggregate_message` and
        :obj:`message_function`.

        Args:
            inputs: [nodes, edges, edge_index]

                - nodes (tf.RaggedTensor): Node embeddings of shape (batch, [N], F)
                - edges (tf.RaggedTensor): Edge or message embeddings of shape (batch, [M], F)
                - edge_index (tf.RaggedTensor): Edge indices referring to nodes of shape (batch, [M], 2)

        Returns:
            tf.RaggedTensor: Updated node embeddings of shape (batch, [N], F)
        """
        nodes, edges, edge_index = inputs
        n_in, n_out = self.lay_gather([nodes, edge_index], **kwargs)
        msg = self.message_function([n_in, n_out, edges], **kwargs)
        pool_n = self.aggregate_message([nodes, msg, edge_index], **kwargs)
        n_new = self.update_nodes([nodes, pool_n], **kwargs)
        return n_new

    def get_config(self):
        """Update config."""
        config = super(MessagePassingBase, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config

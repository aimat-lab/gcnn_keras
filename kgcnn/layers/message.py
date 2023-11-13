import keras_core as ks
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.aggr import AggregateLocalEdges


@ks.saving.register_keras_serializable(package='kgcnn', name='MessagePassingBase')
class MessagePassingBase(ks.layers.Layer):
    r"""Base layer for Message passing type networks. This is a general frame to implement custom message and
    update functions. The idea is to create a subclass of :obj:`MessagePassingBase` and then just implement the methods
    :obj:`message_function` and :obj:`update_nodes`. The pooling or aggregating is handled by built-in
    :obj:`AggregateLocalEdges`.

    Alternatively also :obj:`aggregate_message` could be overwritten.
    The original message passing scheme was proposed by `NMPNN <http://arxiv.org/abs/1704.01212>`__ .
    """

    def __init__(self, pooling_method: str = "scatter_sum", **kwargs):
        r"""Initialize :obj:`MessagePassingBase` layer.

        Args:
            pooling_method (str): Aggregation method of edges. Default is "sum".
        """
        super(MessagePassingBase, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.lay_gather = GatherNodes(concat_axis=None)
        self.lay_pool_default = AggregateLocalEdges(pooling_method=self.pooling_method)
    
    def build(self, input_shape):
        super(MessagePassingBase, self).build(input_shape)

    def message_function(self, inputs, **kwargs):
        r"""Defines the message function, i.e. a method the generates a message from node and edge embeddings at a
        certain depth (not considered here).

        Args:
            inputs: [nodes_in, nodes_out, edge_index]

                - nodes_in (Tensor): Receiving node embeddings of shape ([M], F)
                - nodes_out (Tensor): Sending node embeddings of shape ([M], F)
                - edges (Tensor, optional): Edge or message embeddings of shape ([M], F)

        Returns:
            Tensor: Messages for each edge of shape ([M], F)
        """
        raise NotImplementedError(
            "A method to generate messages must be implemented in sub-class of `MessagePassingBase`.")

    def aggregate_message(self, inputs, **kwargs):
        r"""Pre-defined message aggregation that uses :obj:`AggregateLocalEdges`.

        Args:
            inputs: [nodes, edges, edge_index]

                - nodes (Tensor): Node embeddings of shape ([N], F)
                - edges (Tensor): Edge or message embeddings of shape ([M], F)
                - edge_index (Tensor): Edge indices referring to nodes of shape (2, [M])

        Returns:
            Tensor: Aggregated edge embeddings per node of shape ([N], F)
        """
        return self.lay_pool_default(inputs, **kwargs)

    def update_nodes(self, inputs, **kwargs):
        r"""Defines the update function, i.e. a method that updates the node embeddings from aggregated messages.

        Args:
            inputs: [nodes, node_updates]

                - nodes (Tensor): Node embeddings (from previous step) of shape ([N], F)
                - node_updates (Tensor): Updates for nodes of shape ([N], F)

        Returns:
            Tensor: Updated node embeddings (for next step) of shape ([N], F)
        """
        raise NotImplementedError(
            "A method to update nodes must be implemented in sub-class of `MessagePassingBase`.")

    def call(self, inputs, **kwargs):
        r"""Pre-implemented standard message passing scheme using :obj:`update_nodes`, :obj:`aggregate_message` and
        :obj:`message_function`.

        Args:
            inputs: [nodes, edges, edge_index]

                - nodes (Tensor): Node embeddings of shape ([N], F)
                - edges (Tensor, optional): Edge or message embeddings of shape ([M], F)
                - edge_index (Tensor): Edge indices referring to nodes of shape (2, [M])

        Returns:
            Tensor: Updated node embeddings of shape ([N], F)
        """
        if len(inputs) == 2:
            nodes, edge_index = inputs
            edges = None
        else:
            nodes, edges, edge_index = inputs

        n_in, n_out = self.lay_gather([nodes, edge_index], **kwargs)

        if edges is None:
            msg = self.message_function([n_in, n_out], **kwargs)
        else:
            msg = self.message_function([n_in, n_out, edges], **kwargs)

        pool_n = self.aggregate_message([nodes, msg, edge_index], **kwargs)
        n_new = self.update_nodes([nodes, pool_n], **kwargs)
        return n_new

    def get_config(self):
        """Update config."""
        config = super(MessagePassingBase, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config

import keras as ks
from keras import ops
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

    def __init__(self,
                 pooling_method: str = "scatter_sum",
                 use_id_tensors: int = None,
                 **kwargs):
        r"""Initialize :obj:`MessagePassingBase` layer.

        Args:
            pooling_method (str): Aggregation method of edges. Default is "sum".
            use_id_tensors (int): Whether :obj:`call` receives graph ID information, which is passed onto message and
                aggregation function. Specifies the number of additional tensors.
        """
        super(MessagePassingBase, self).__init__(**kwargs)
        self.pooling_method = pooling_method
        self.lay_gather = GatherNodes(concat_axis=None)
        self.use_id_tensors = use_id_tensors
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
        if self.use_id_tensors is not None:
            ids = inputs[-int(self.use_id_tensors):]
            inputs = inputs[:-int(self.use_id_tensors)]
        else:
            ids = []

        if len(inputs) == 2:
            nodes, edge_index = inputs[:2]
            edges = None
        else:
            nodes, edges, edge_index = inputs[:3]

        n_in, n_out = self.lay_gather([nodes, edge_index], **kwargs)

        if edges is None:
            msg = self.message_function([n_in, n_out] + ids, **kwargs)
        else:
            msg = self.message_function([n_in, n_out, edges] + ids, **kwargs)

        pool_n = self.aggregate_message([nodes, msg, edge_index], **kwargs)

        n_new = self.update_nodes([nodes, pool_n] + ids, **kwargs)
        return n_new

    def get_config(self):
        """Update config."""
        config = super(MessagePassingBase, self).get_config()
        config.update({"pooling_method": self.pooling_method, "use_id_tensors": self.use_id_tensors})
        return config


class MatMulMessages(ks.layers.Layer):
    r"""Linear transformation of edges or messages, i.e. matrix multiplication.

    The message dimension must be suitable for matrix multiplication. The actual matrix is not a trainable weight of
    this layer but passed as input.
    This was proposed by `NMPNN <http://arxiv.org/abs/1704.01212>`__ .
    For each node or edge :math:`i` the output is given by:

    .. math::

        x_i' = \mathbf{A_i} \; x_i
    """

    def __init__(self, **kwargs):
        """Initialize layer."""
        super(MatMulMessages, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        super(MatMulMessages, self).build(input_shape)

    def call(self, inputs, **kwargs):
        r"""Forward pass.

        Args:
            inputs (list): [mat, edges]

                - mat (Tensor): Transformation matrix for each message of shape ([M], F', F).
                - edges (Tensor): Edge embeddings or messages ([M], F)

        Returns:
            Tensor: Transformation of messages by matrix multiplication of shape (batch, [M], F')
        """
        mat, e = inputs
        e_e = ops.expand_dims(e, axis=1)
        out = ops.sum(mat*e_e, axis=2)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(MatMulMessages, self).get_config()
        return config

.. _implementation:
   :maxdepth: 3

Models
======

Representation
--------------

The most frequent usage for graph convolutions is either node or graph classification. As for their size, either a single large graph, e.g. citation network or small (batched) graphs like molecules have to be considered.
Graphs can be represented by an index list of connections plus feature information. Typical quantities in tensor format to describe a graph are listed below.

* ``nodes``: Node-list of shape ``(batch, [N], F)`` where ``N`` is the number of nodes and ``F`` is the node feature dimension.
* ``edges``: Edge-list of shape ``(batch, [M], F)`` where ``M`` is the number of edges and ``F`` is the edge feature dimension.
* ``indices``: Connection-list of shape ``(batch, [M], 2)`` where ``M`` is the number of edges. The indices denote a connection of incoming or receiving node ``i`` and outgoing or sending node ``j`` as ``(i, j)``.
* ``state``: Graph state information of shape ``(batch, F)`` where ``F`` denotes the feature dimension.

A major issue for graphs is their flexible size and shape, when using mini-batches. Here, for a graph implementation in the spirit of keras, the batch dimension should be kept also in between layers. This is realized by using ``RaggedTensor`` .

Input
-----

Graph tensors for edge-indices or attributes for multiple graphs is passed to the model in form of ragged tensors
of shape ``(batch, None, Dim)`` where ``Dim`` denotes a fixed feature or index dimension.
Such a ragged tensor has ``ragged_rank=1`` with one ragged dimension indicated by ``None`` and is build from a value plus partition tensor.
For example, the graph structure is represented by an index-list of shape ``(batch, None, 2)`` with index of incoming or receiving node ``i`` and outgoing or sending node ``j`` as ``(i, j)``.
Note, an additional edge with ``(j, i)`` is required for undirected graphs.
A ragged constant can be easily created and passed to a model::


    import tensorflow as tf
    import numpy as np
    idx = [[[0, 1], [1, 0]], [[0, 1], [1, 2], [2, 0]], [[0, 0]]]  # batch_size=3
    # Get ragged tensor of shape (3, None, 2)
    print(tf.ragged.constant(idx, ragged_rank=1, inner_shape=(2, )).shape)
    print(tf.RaggedTensor.from_row_lengths(np.concatenate(idx), [len(i) for i in idx]).shape)


Model
-----

Models can be set up in a functional way. Example message passing from fundamental operations::

    import tensorflow.keras as ks
    from kgcnn.layers.gather import GatherNodes
    from kgcnn.layers.modules import Dense, LazyConcatenate  # ragged support
    from kgcnn.layers.pooling import PoolingLocalMessages, PoolingNodes

    n = ks.layers.Input(shape=(None, 3), name='node_input', dtype="float32", ragged=True)
    ei = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)

    n_in_out = GatherNodes()([n, ei])
    node_messages = Dense(10, activation='relu')(n_in_out)
    node_updates = PoolingLocalMessages()([n, node_messages, ei])
    n_node_updates = LazyConcatenate(axis=-1)([n, node_updates])
    n_embedd = Dense(1)(n_node_updates)
    g_embedd = PoolingNodes()(n_embedd)

    message_passing = ks.models.Model(inputs=[n, ei], outputs=g_embedd)

or via sub-classing of the message passing base layer. Where only ``message_function`` and ``update_nodes`` must be implemented::

    from kgcnn.layers.conv.message import MessagePassingBase
    from kgcnn.layers.modules import Dense, LazyAdd

    class MyMessageNN(MessagePassingBase):
      def __init__(self, units, **kwargs):
        super(MyMessageNN, self).__init__(**kwargs)
        self.dense = Dense(units)
        self.add = LazyAdd(axis=-1)

      def message_function(self, inputs, **kwargs):
        n_in, n_out, edges = inputs
        return self.dense(n_out)

      def update_nodes(self, inputs, **kwargs):
        nodes, nodes_update = inputs
        return self.add([nodes, nodes_update])


.. _implementation:
   :maxdepth: 3

Implementation
==============

Representation
--------------

The most frequent usage for graph convolutions is either node or graph classification. As for their size, either a single large graph, e.g. citation network or small (batched) graphs like molecules have to be considered.
Graphs can be represented by an index list of connections plus feature information. Typical quantities in tensor format to describe a graph are listed below.

* ``nodes``: Nodelist of shape ``(batch, N, F)`` where ``N`` is the number of nodes and ``F`` is the node feature dimension.
* ``edges``: Edge-list of shape ``(batch, M, F)`` where ``M`` is the number of edges and ``Fe`` is the edge feature dimension.
* ``indices``: Connection-list of shape ``(batch, M, 2)`` where ``M`` is the number of edges. The indices denote a connection of incoming i and outgoing j node as ``(i,j)``.
* ``state``: Graph state information of shape ``(batch, F)`` where ``F`` denotes the feature dimension.

A major issue for graphs is their flexible size and shape, when using mini-batches. Here, for a graph implementation in the spirit of keras, the batch dimension should be kept also in between layers. This is realized by using ``RaggedTensor``.

Input
-----

Here, for ragged tensors, the nodelist of shape ``(batch, None, F)`` and edgelist of shape ``(batch, None, F')`` have one ragged dimension ``(None, )``.
The graph structure is represented by an index-list of shape ``(batch, None, 2)`` with index of incoming ``i`` and outgoing ``j`` node as ``(i, j)``.
The first index of incoming node ``i`` is usually expected to be sorted for faster pooling operations, but can also be unsorted (see layer arguments).
Furthermore, the graph is directed, so an additional edge with ``(j, i)`` is required for undirected graphs.
A ragged constant can be directly obtained from a list of numpy arrays: ``tf.ragged.constant(indices, ragged_rank=1, inner_shape=(2, ))`` which yields shape ``(batch, None, 2)``.

Model
-----

Models can be set up in a functional way. Example message passing from fundamental operations::

    import tensorflow.keras as ks
    from kgcnn.layers.gather import GatherNodes
    from kgcnn.layers.keras import Dense, Concatenate  # ragged support
    from kgcnn.layers.pool.pooling import PoolingLocalMessages, PoolingNodes

    n = ks.layers.Input(shape=(None, 3), name='node_input', dtype="float32", ragged=True)
    ei = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)

    n_in_out = GatherNodes()([n, ei])
    node_messages = Dense(10, activation='relu')(n_in_out)
    node_updates = PoolingLocalMessages()([n, node_messages, ei])
    n_node_updates = Concatenate(axis=-1)([n, node_updates])
    n_embedd = Dense(1)(n_node_updates)
    g_embedd = PoolingNodes()(n_embedd)

    message_passing = ks.models.Model(inputs=[n, ei], outputs=g_embedd)

or via sub-classing of the message passing base layer. Where only ``message_function`` and ``update_nodes`` must be implemented::

    from kgcnn.layers.conv.message import MessagePassingBase
    from kgcnn.layers.keras import Dense, Add

    def MyMessageNN(MessagePassingBase):

        def __init__(self, units, **kwargs):
            super(MyMessageNN, self).__init__(**kwargs)
            self.dense = Dense(units)
            self.add = Add(axis=-1)

        def message_function(self, inputs, **kwargs):
            n_in, n_out, edges = inputs
            return self.dense(n_out)

        def update_nodes(self, inputs, **kwargs):
            nodes, nodes_update = inputs
            return self.add([nodes, nodes_update])


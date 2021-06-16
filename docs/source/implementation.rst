.. _implementation:
   :maxdepth: 3

Implementation
==============

Representation
--------------

The most frequent usage for graph convolutions is either node or graph classifaction. As for their size, either a single large graph, e.g. citation network or small (batched) graphs like molecules have to be considered. 
Graphs can be represented by an index list of connections plus feature information. Typical quantities in tensor format to describe a graph are listed below.

* ``nodes``: Nodelist of shape ``(batch, N, F)`` where ``N`` is the number of nodes and ``F`` is the node feature dimension.
* ``edges``: Edgelist of shape ``(batch, M, F)`` where ``M`` is the number of edges and ``Fe`` is the edge feature dimension.
* ``indices``: Connectionlist of shape ``(batch, M, 2)`` where ``M`` is the number of edges. The indices denote a connection of incoming i and outgoing j node as ``(i,j)``.
* ``state``: Graph state information of shape ``(batch, F)`` where ``F`` denotes the feature dimension.

A major issue for graphs is their flexible size and shape, when using mini-batches. Here, for a graph implementation in the spirit of keras, the batch dimension should be kept also in between layers. This is realized by using ``RaggedTensor``.
 
Note: At the moment, most layers support also a disjoint representation of flatten values plus graph-id tensor ``[values, partition]`` in place of the ``RaggedTensor`` for comparison purposes. 
However, this will likely be removed in future versions, as ``RaggedTensor`` is intended be the only tensor representation passed to and within the model.

Input
-----

In order to input batched tensors of variable length with keras, either zero-padding plus masking or ragged and sparse tensors can be used. Morover for more flexibility, a dataloader from `tf.keras.utils.Sequence` is often used to input disjoint graph representations. Tools for converting numpy or scipy arrays are found in `utils <https://github.com/aimat-lab/gcnn_keras/tree/master/kgcnn/utils>`_.

Here, for ragged tensors, the nodelist of shape ``(batch, None, F)`` and edgelist of shape ``(batch, None, Fe)`` have one ragged dimension ``(None, )``.
The graph structure is represented by an indexlist of shape ``(batch, None, 2)`` with index of incoming ``i`` and outgoing ``j`` node as ``(i, j)``. 
The first index of incoming node ``i`` is usually expected to be sorted for faster pooling opertions, but can also be unsorted (see layer arguments). Furthermore the graph is directed, so an additional edge with ``(j, i)`` is required for undirected graphs. A ragged constant can be directly obtained from a list of numpy arrays: ``tf.ragged.constant(indices,ragged_rank=1,inner_shape=(2,))`` which yields shape ``(batch, None, 2)``.

Model
-----

Models can be set up in a functional. Example message passing from fundamental operations::

   import tensorflow.keras as ks
   from kgcnn.layers.gather import GatherNodes
   from kgcnn.layers.keras import Dense, Concatenate  # ragged support
   from kgcnn.layers.pooling import PoolingLocalMessages, PoolingNodes
   
   n = ks.layers.Input(shape=(None, 3), name='node_input', dtype="float32", ragged=True)
   ei = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)
   
   n_in_out = GatherNodes()([n, ei])
   node_messages = Dense(10, activation='relu')(n_in_out)
   node_updates = PoolingLocalMessages()([n, node_messages, ei])
   n_node_updates = Concatenate(axis=-1)([n, node_updates])
   n_embedd = Dense(1)(n_node_updates)
   g_embedd = PoolingNodes()(n_embedd)
   
   message_passing = ks.models.Model(inputs=[n, ei], outputs=g_embedd)

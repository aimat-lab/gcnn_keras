.. _implementation:
   :maxdepth: 3

Implementation
==============

Representation
--------------

The most frequent usage for graph convolutions is either node or graph classifaction. As for their size, either a single large graph, e.g. citation network or small (batched) graphs like molecules have to be considered. 
Graphs can be represented by a connection index list plus feature information. Typical quantities in tensor format to describe a graph are listed below.

* ``n``: Nodelist of shape ``([batch],N,F)`` where ``N`` is the number of nodes and ``F`` is the node feature dimension.
* ``e``: Edgelist of shape ``([batch],M,Fe)`` where ``M`` is the number of edges and ``Fe`` is the edge feature dimension.
* ``m``: Connectionlist of shape ``([batch],M,2)`` where ``M`` is the number of edges. The indices denote a connection of incoming i and outgoing j node as ``(i,j)``.
* ``u``: Graph state information of shape ``([batch],F)`` where ``F`` denotes the feature dimension.
 
A major issue for graphs is their flexible size and shape, when using mini-batches. Here, for a graph implementation in the spirit of keras, the batch dimension should be kept also in between layes. This is realized by using ragged tensors. A complete set of layers that work solemnly with ragged tensors is given in `disjoint <https://github.com/aimat-lab/gcnn_keras/tree/master/kgcnn/layers/ragged>`_.

Many graph implementations use also a `disjoint <https://github.com/aimat-lab/gcnn_keras/tree/master/kgcnn/layers/disjoint>`_ representation and 
`sparse <https://github.com/aimat-lab/gcnn_keras/tree/master/kgcnn/layers/sparse>`_ or 
`padded <https://github.com/aimat-lab/gcnn_keras/tree/master/kgcnn/layers/padded>`_ tensors.

Input
-----

In order to input batched tensors of variable length with keras, either zero-padding plus masking or ragged and sparse tensors can be used. Morover for more flexibility, a dataloader from `tf.keras.utils.Sequence` is often used to input disjoint graph representations. Tools for converting numpy or scipy arrays are found in [utils](kgcnn/utils).

Here, for ragged tensors, the nodelist of shape ``(batch,None,F)`` and edgelist of shape ``(batch,None,Fe)`` have one ragged dimension ``(None,)``.
The graph structure is represented by an indexlist of shape ``(batch,None,2)`` with index of incoming ``i`` and outgoing ``j`` node as ``(i,j)``. 
The first index of incoming node ``i`` is usually expected to be sorted for faster pooling opertions, but can also be unsorted (see layer arguments). Furthermore the graph is directed, so an additional edge with ``(j,i)`` is required for undirected graphs. A ragged constant can be directly obtained from a list of numpy arrays: ``tf.ragged.constant(indices,ragged_rank=1,inner_shape=(2,))`` which yields shape ``(batch,None,2)``.

Model
-----

Models can be set up in a functional. Example message passing from fundamental operations::

   import tensorflow as tf
   import tensorflow.keras as ks
   from kgcnn.layers.ragged.gather import GatherNodes
   from kgcnn.layers.ragged.conv import DenseRagged  # Will most likely be supported by keras.Dense in the future
   from kgcnn.layers.ragged.pooling import PoolingEdgesPerNode

   feature_dim = 10
   n = ks.layers.Input(shape=(None,feature_dim),name='node_input',dtype ="float32",ragged=True)
   ei = ks.layers.Input(shape=(None,2),name='edge_index_input',dtype ="int64",ragged=True)

   n_in_out = GatherNodes()([n,ei])
   node_messages = DenseRagged(feature_dim)(n_in_out)
   node_updates = PoolingEdgesPerNode()([n,node_messages,ei])
   n_node_updates = ks.layers.Concatenate(axis=-1)([n,node_updates])
   n_embedd = DenseRagged(feature_dim)(n_node_updates)

   message_passing = ks.models.Model(inputs=[n,ei], outputs=n_embedd)

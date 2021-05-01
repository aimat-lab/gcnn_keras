import tensorflow as tf
import tensorflow.keras as ks

class CastRaggedToDisjointSparseAdjacency(tf.keras.layers.Layer):
    """
    Layer to cast RaggedTensor graph representation to a single Sparse tensor in disjoint representation.

    This includes edge_indices and adjacency matrix entries. The Sparse tensor is simply the adjacency matrix.

    Args:
        node_indexing (str): If edge_indices refer to sample- or batch-wise indexing. Default is 'sample'.
        is_sorted (bool): If the edge_indices are sorted for first ingoing index. Default is False.
        ragged_validate (bool): To validate the ragged output tensor. Default is False.
        **kwargs
    """

    def __init__(self,
                 node_indexing="sample",
                 is_sorted=False,
                 ragged_validate=False,
                 **kwargs):
        """Initialize layer."""
        super(CastRaggedToDisjointSparseAdjacency, self).__init__(**kwargs)
        self.ragged_validate = ragged_validate
        self.is_sorted = is_sorted
        self.node_indexing = node_indexing
        self._supports_ragged_inputs = True

    def build(self, input_shape):
        """Build layer."""
        super(CastRaggedToDisjointSparseAdjacency, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            Inputs list of [nodes,edge_index,edges]

            - nodes (tf.ragged): Node feature tensor of shape (batch,None,F)
            - edge_index (tf.ragged): Ragged edge_indices of shape (batch,None,2)
            - edges (tf.ragged): Edge feature ragged tensor of shape (batch,None,1)

        Returns:
            tf.sparse: Sparse disjoint matrix of shape (batch*None,batch*None)
        """
        nod, edgeind, ed = inputs

        if self.node_indexing == 'batch':
            shiftind = edgeind.values
        elif self.node_indexing == 'sample':
            shift1 = edgeind.values
            shift2 = tf.expand_dims(tf.repeat(nod.row_splits[:-1], edgeind.row_lengths()), axis=1)
            shiftind = shift1 + tf.cast(shift2, dtype=shift1.dtype)
        else:
            raise TypeError("Unknown index convention, use: 'sample', 'batch', ...")

        indexlist = shiftind
        valuelist = ed.values

        if not self.is_sorted:
            # Sort per outgoing
            batch_order = tf.argsort(indexlist[:, 1], axis=0, direction='ASCENDING')
            indexlist = tf.gather(indexlist, batch_order, axis=0)
            valuelist = tf.gather(valuelist, batch_order, axis=0)
            # Sort per ingoing node
            node_order = tf.argsort(indexlist[:, 0], axis=0, direction='ASCENDING', stable=True)
            indexlist = tf.gather(indexlist, node_order, axis=0)
            valuelist = tf.gather(valuelist, node_order, axis=0)

        indexlist = tf.cast(indexlist, dtype=tf.int64)
        dense_shape = tf.concat([tf.shape(nod.values)[0:1], tf.shape(nod.values)[0:1]], axis=0)
        dense_shape = tf.cast(dense_shape, dtype=tf.int64)
        out = tf.sparse.SparseTensor(indexlist, valuelist[:, 0], dense_shape)

        return out

    def get_config(self):
        """Update layer config."""
        config = super(CastRaggedToDisjointSparseAdjacency, self).get_config()
        config.update({"ragged_validate": self.ragged_validate})
        config.update({"node_indexing": self.node_indexing})
        config.update({"is_sorted": self.is_sorted})
        return config


class PoolingAdjacencyMatmul(ks.layers.Layer):
    r"""
    Layer for pooling of node features by multiplying with sparse adjacency matrix. Which gives $A n$.
    The node features needs to be flatten for a disjoint representation.

    Args:
        pooling_method : tf.function to pool all nodes compatible with dense tensors.
        **kwargs
    """

    def __init__(self,
                 pooling_method="sum",
                 **kwargs):
        """Initialize layer."""
        super(PoolingAdjacencyMatmul, self).__init__(**kwargs)
        self.pooling_method = pooling_method

    def build(self, input_shape):
        """Build layer."""
        super(PoolingAdjacencyMatmul, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            list: [nodes, adjacency]

            - nodes (tf.tensor): Flatten node features of shape (batch*None,F)
            - adjacency (tf.sparse): SparseTensor of the adjacency matrix of shape (batch*None,batch*None)

        Returns:
            features (tf.tensor): Pooled node features of shape (batch,F)
        """
        node, adj = inputs
        out = tf.sparse.sparse_dense_matmul(adj, node)

        return out

    def get_config(self):
        """Update layer config."""
        config = super(PoolingAdjacencyMatmul, self).get_config()
        config.update({"pooling_method": self.pooling_method})
        return config
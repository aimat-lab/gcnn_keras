import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as ksb

from kgcnn.utils.partition import _change_edge_tensor_indexing_by_row_partition
from kgcnn.utils.soft import segment_softmax

class PoolingLocalEdgesAttention(ks.layers.Layer):
    r"""
    Pooling all edges or edgelike features per node, corresponding to node assigned by edge indices.
    Uses attention for pooling. i.e.  $n_i =  \sum_j \alpha_{ij} e_ij $
    The attention is computed via: $\alpha_ij = softmax(a_ij)$ and from the attention coefficients $a_ij$.
    The attention coefficients must be computed beforehand by edge features or by $\sigma( W n_i || W n_j)$

    If graphs indices were in 'sample' mode, the indices must be corrected for disjoint graphs.
    Apply e.g. segment_mean for index[0] incoming nodes.
    Important: edge_index[:,0] are sorted for segment-operation.

    Args:
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,
                 node_indexing="batch",
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 **kwargs):
        """Initialize layer."""
        super(PoolingLocalEdgesAttention, self).__init__(**kwargs)
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.partition_type = partition_type

    def build(self, input_shape):
        """Build layer."""
        super(PoolingLocalEdgesAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, node_partition, edges, attention, edge_partition, edge_indices]

            - node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
              only required for target shape, so that pooled tensor has same shape!
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edges (tf.tensor): Flatten edge features or node_j for edge_ij or node_i||node_j of shape (batch*None,F)
            - attention (tf.tensor): Attention coefficients to compute the attention from, must be shape (batch*None,1)
              and match the edges, i.e. have same first dimension and node assignment a(i,j) match e(i,j)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_indices (tf.tensor): Flatten index list tensor of shape (batch*None,2)
              pooling is done according to first index i from edge index pair (i,j)

        Returns:
            features (tf.tensor): Flatten feature tensor of pooled edge attentions for each node.
            The size will match the flatten node tensor.
            Output shape is (batch*None, F).
        """
        nod, node_part, edge, attention, edge_part, edgeind = inputs

        shiftind = _change_edge_tensor_indexing_by_row_partition(edgeind, node_part, edge_part,
                                                                 partition_type_node=self.partition_type,
                                                                 partition_type_edge=self.partition_type,
                                                                 to_indexing='batch',
                                                                 from_indexing=self.node_indexing)

        nodind = shiftind[:, 0]  # Pick first index eg. ingoing
        dens = edge
        ats = attention
        if not self.is_sorted:
            # Sort edgeindices
            node_order = tf.argsort(nodind, axis=0, direction='ASCENDING', stable=True)
            nodind = tf.gather(nodind, node_order, axis=0)
            dens = tf.gather(dens, node_order, axis=0)
            ats = tf.gather(ats, node_order, axis=0)

        # Apply segmented softmax
        ats = segment_softmax(ats,nodind)
        get = dens*ats
        get = tf.math.segment_sum(get,nodind)

        if self.has_unconnected:
            # Need to fill tensor since the maximum node may not be also in pooled
            # Does not happen if all nodes are also connected
            pooled_index = tf.range(tf.shape(get)[0])  # tf.unique(nodind)
            outtarget_shape = (tf.shape(nod, out_type=nodind.dtype)[0], ks.backend.int_shape(dens)[-1])
            get = tf.scatter_nd(ks.backend.expand_dims(pooled_index, axis=-1), get, outtarget_shape)

        return get

    def get_config(self):
        """Update layer config."""
        config = super(PoolingLocalEdgesAttention, self).get_config()
        config.update({"pooling_method": self.pooling_method,
                       "is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type})
        return config




class AttentionHead(ks.layers.Layer):
    r"""Computes the attention head.

    If graphs indices were in 'sample' mode, the indices must be corrected for disjoint graphs.

    Args:
        node_indexing (str): Indices refering to 'sample' or to the continous 'batch'.
                             For disjoint representation 'batch' is default.
        is_sorted (bool): If the edge indices are sorted for first ingoing index. Default is False.
        has_unconnected (bool): If unconnected nodes are allowed. Default is True.
        partition_type (str): Partition tensor type to assign nodes/edges to batch. Default is "row_length".
        **kwargs
    """

    def __init__(self,
                 node_indexing="batch",
                 is_sorted=False,
                 has_unconnected=True,
                 partition_type="row_length",
                 **kwargs):
        """Initialize layer."""
        super(AttentionHead, self).__init__(**kwargs)
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.partition_type = partition_type



    def build(self, input_shape):
        """Build layer."""
        super(AttentionHead, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, node_partition, edges, edge_partition, edge_indices]

            - node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
              only required for target shape, so that pooled tensor has same shape!
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edges (tf.tensor): Flatten edge feature tensor of shape (batch*None,F)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_indices (tf.tensor): Flatten index list tensor of shape (batch*None,2)

        Returns:
            features (tf.tensor): Flatten feature tensor of pooled edge attentions for each node.
            The size will match the flatten node tensor.
            Output shape is (batch*None, F).
        """
        pass


    def get_config(self):
        """Update layer config."""
        config = super(AttentionHead, self).get_config()
        config.update({"is_sorted": self.is_sorted,
                       "has_unconnected": self.has_unconnected,
                       "node_indexing": self.node_indexing,
                       "partition_type": self.partition_type})
        return config
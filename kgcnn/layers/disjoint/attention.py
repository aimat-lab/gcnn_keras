import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as ksb

from kgcnn.utils.partition import _change_partition_type, _change_edge_tensor_indexing_by_row_partition


class AttentionNodes(ks.layers.Layer):
    """
    Pooling all edges or edgelike features per node, corresponding to node assigned by edge indices.

    If graphs indices were in 'sample' mode, the indices must be corrected for disjoint graphs.
    Apply e.g. segment_mean for index[0] incoming nodes.
    Important: edge_index[:,0] are sorted for segment-operation.

    Args:
        pooling_method (str): Pooling method to use i.e. segement_function. Default is 'segment_mean'.
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
        super(AttentionNodes, self).__init__(**kwargs)
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        self.node_indexing = node_indexing
        self.partition_type = partition_type



    def build(self, input_shape):
        """Build layer."""
        super(AttentionNodes, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """Forward pass.

        Args:
            inputs (list): of [node, node_partition, edges, edge_partition, edge_indices]

            - node (tf.tensor): Flatten node feature tensor of shape (batch*None,F)
              only required for target shape, so that pooled tensor has same shape!
            - node_partition (tf.tensor): Row partition for nodes. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of nodes to each graph in batch.
              Default is row_length of shape (batch,)
            - edges (tf.tensor): Flatten edge feature or Node_i||Node_j or edge_ij||Node_i||node_j tensor
              of shape (batch*None,F)
            - edge_partition (tf.tensor): Row partition for edge. This can be either row_length, value_rowids,
              row_splits. Yields the assignment of edges to each graph in batch.
              Default is row_length of shape (batch,)
            - edge_indices (tf.tensor): Flatten index list tensor of shape (batch*None,2)

        Returns:
            features (tf.tensor): Flatten feature tensor of pooled edge features for each node.
            The size will match the flatten node tensor.
            Output shape is (batch*None, F).
        """
        nod, node_part, edge, edge_part, edgeind = inputs

        shiftind = _change_edge_tensor_indexing_by_row_partition(edgeind, node_part, edge_part,
                                                                 partition_type_node=self.partition_type,
                                                                 partition_type_edge=self.partition_type,
                                                                 to_indexing='batch',
                                                                 from_indexing=self.node_indexing)



        out = get
        return out
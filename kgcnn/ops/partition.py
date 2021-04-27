import tensorflow as tf


@tf.function
def kgcnn_ops_change_edge_tensor_indexing_by_row_partition(edge_index, part_node, part_edge,
                                                           partition_type_node,
                                                           partition_type_edge,
                                                           from_indexing='sample',
                                                           to_indexing='batch',
                                                           axis=1):
    """Change the edge index tensor indexing between per graph and per batch assignment. Batch assignment is equivalent
    to disjoint representation. To change indices, the row partition of edge and node tensor must be known.

    Args:
        edge_index (tf.tensor): Edge indices of shape (batch*None, 2)
        part_node (tf.tensor):  Node partition tensor.
        part_edge (tf.tensor): Edge partition tensor.
        partition_type_node (str:) Node type of partition, can be either 'row_splits', 'row_length' or 'value_rowids'
        partition_type_edge (str): Edge type of partition, can be either 'row_splits', 'row_length' or 'value_rowids'
        from_indexing (str): Source index scheme
        to_indexing (str): Target index scheme
        axis (int): Axis along which to shift indices. Default is 1.

    Returns:
        tf.tensor: Index tensor of shifted indices.
    """
    if to_indexing == from_indexing:
        # Nothing to do here
        return edge_index

    # we need node row_splits
    nod_splits = None
    if partition_type_node == "row_length":
        node_len = part_node
        nod_splits = tf.cumsum(node_len, exclusive=True) # Can be without total sum
    elif partition_type_node == "row_splits":
        nod_splits = part_node
    elif partition_type_node == "value_rowids":
        node_len = tf.math.segment_sum(tf.ones_like(part_node), part_node)
        nod_splits = tf.cumsum(node_len, exclusive=True) # Can be without total sum
    else:
        raise TypeError("Unknown node partition scheme, use: 'value_rowids', 'row_splits', row_length")

    # we need edge value_rowids
    edge_ids = None
    if partition_type_edge == "row_length":
        edge_len = part_edge
        edge_ids = tf.repeat(tf.range(tf.shape(edge_len)[0]), edge_len)
    elif partition_type_edge == "row_splits":
        edge_len = part_edge[1:] - part_edge[:-1]
        edge_ids = tf.repeat(tf.range(tf.shape(edge_len)[0]), edge_len)
    elif partition_type_edge == "value_rowids":
        edge_ids = part_edge
    else:
        raise TypeError("Unknown edge partition scheme, use: 'value_rowids', 'row_splits', row_length")

    # Just gather the splits i.e. index offset for each graph id
    shift_index = tf.gather(nod_splits, edge_ids)

    # Expand dimension to broadcast to indices for suitable axis
    # The shift_index is always 1D tensor.
    for i in range(1,axis+1):
        shift_index = tf.expand_dims(shift_index,axis=i)

    # Add or substract batch offset from index tensor
    if to_indexing == 'batch' and from_indexing == 'sample':
        indexlist = edge_index + tf.cast(shift_index, dtype=edge_index.dtype)
    elif to_indexing == 'sample' and from_indexing == 'batch':
        indexlist = edge_index - tf.cast(shift_index, dtype=edge_index.dtype)
    else:
        raise TypeError("Unknown index change, use: 'sample', 'batch', ...")

    out = indexlist
    return out


@tf.function
def kgcnn_ops_change_partition_type(in_partition, in_partition_type, out_partition_type):
    """Switch between partition types.

    Args:
        in_partition (tf.tensor): Row partition tensor
        in_partition_type (str): Source partition type, can be either 'row_splits', 'row_length' or 'value_rowids'
        out_partition_type (str): Target partition type, can be either 'row_splits', 'row_length' or 'value_rowids'

    Returns:
        out_partition (tf.tensor): Row partition tensor of target type.
    """
    if in_partition_type == out_partition_type:
        # Do nothing here
        out_partition = in_partition
    elif in_partition_type == "row_length" and out_partition_type == "row_splits":
        # We need ex. (1,2,3) -> (0,1,3,6)
        out_partition = tf.pad(tf.cumsum(in_partition), [[1, 0]])
    elif in_partition_type == "row_splits" and out_partition_type == "row_length":
        # Matches length if (0,1,3,6) -> (1,2,3)
        out_partition = in_partition[1:] - in_partition[:-1]
    elif in_partition_type == "row_length" and out_partition_type == "value_rowids":
        # May cast to dtype = tf.int32 here
        out_partition = tf.repeat(tf.range(tf.shape(in_partition)[0]), in_partition)
    elif in_partition_type == "value_rowids" and out_partition_type == "row_length":
        out_partition = tf.math.segment_sum(tf.ones_like(in_partition), in_partition)
    elif in_partition_type == "value_rowids" and out_partition_type == "row_splits":
        # Get row_length
        part_sum = tf.math.segment_sum(tf.ones_like(in_partition), in_partition)
        out_partition = tf.pad(tf.cumsum(part_sum), [[1, 0]])
    elif in_partition_type == "row_splits" and out_partition_type == "value_rowids":
        # Get row_length
        part_sum = in_partition[1:] - in_partition[:-1]
        out_partition = tf.repeat(tf.range(tf.shape(part_sum)[0]), part_sum)
    else:
        raise TypeError("Unknown partition scheme, use: 'value_rowids', 'row_splits', row_length")

    return out_partition

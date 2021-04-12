import tensorflow as tf


@tf.function
def _change_edge_tensor_indexing_by_row_partition(edge_index, part_node, part_edge, partition_type,
                                                  from_indexing='sample',
                                                  to_indexing='batch'):
    """Change the edge index tensor indexing between per graph and per batch assignment. Batch assignment is equivalent
    to disjoint representation. To change indices, the row partition of edge and node tensor must be known.

    Args:
        edge_index (tf.tensor): Edge indices of shape (batch*None, 2)
        part_node (tf.tensor):  Node partition tensor.
        part_edge (tf.tensor): Edge partition tensor.
        partition_type (str): Type of partition, can be either 'row_splits', 'row_length' or 'value_rowids'
        from_indexing (str): Source index scheme
        to_indexing (str): Target index scheme

    Returns:

    """
    # if self.node_indexing == 'batch':
    #     indexlist = edge_index
    # elif self.node_indexing == 'sample':
    #     shift1 = edge_index
    #     if self.partition_type == "row_length":
    #         edge_len = edge_part
    #         node_len = node_part
    #         shift2 = tf.expand_dims(tf.repeat(tf.cumsum(node_len, exclusive=True), edge_len), axis=1)
    #     elif self.partition_type == "row_splits":
    #         edge_len = edge_part[1:] - edge_part[:-1]
    #         shift2 = tf.expand_dims(tf.repeat(node_part[:-1], edge_len), axis=1)
    #     elif self.partition_type == "value_rowids":
    #         edge_len = tf.math.segment_sum(tf.ones_like(edge_part), edge_part)
    #         node_len = tf.math.segment_sum(tf.ones_like(node_part), node_part)
    #         shift2 = tf.expand_dims(tf.repeat(tf.cumsum(node_len, exclusive=True), edge_len), axis=1)
    #     else:
    #         raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...")
    #     indexlist = shift1 + tf.cast(shift2, dtype=shift1.dtype)
    # else:
    #     raise TypeError("Unknown index convention, use: 'sample', 'batch', ...")

    if to_indexing != from_indexing:
        # splits[1:] - splits[:-1]
        if partition_type == "row_length":
            shift_index = tf.expand_dims(tf.repeat(tf.cumsum(part_node, exclusive=True), part_edge), axis=1)
        elif partition_type == "row_splits":
            edge_len = part_edge[1:] - part_edge[:-1]
            shift_index = tf.expand_dims(tf.repeat(part_node[:-1], edge_len), axis=1)
        elif partition_type == "value_rowids":
            node_len = tf.math.segment_sum(tf.ones_like(part_node), part_node)
            edge_len = tf.math.segment_sum(tf.ones_like(part_edge), part_edge)
            shift_index = tf.expand_dims(tf.repeat(tf.cumsum(node_len, exclusive=True), edge_len), axis=1)
        else:
            raise TypeError("Unknown partition scheme, use: 'row_length', 'row_splits', ...")

        # Add or substract batch offset from index tensor
    if to_indexing == 'batch' and from_indexing == 'sample':
        indexlist = edge_index + tf.cast(shift_index, dtype=edge_index.dtype)
    elif to_indexing == 'sample' and from_indexing == 'batch':
        indexlist = edge_index - tf.cast(shift_index, dtype=edge_index.dtype)
    elif to_indexing == 'sample' and from_indexing == 'sample':
        indexlist = edge_index
    elif to_indexing == 'batch' and from_indexing == 'batch':
        indexlist = edge_index
    else:
        raise TypeError("Unknown index change, use: 'sample', 'batch', ...")

    out = indexlist
    return out


@tf.function
def _change_partition_type(in_partition, in_partition_type, out_partition_type):
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
        # @TODO: Can just use tf.scatter
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

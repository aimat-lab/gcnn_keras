import tensorflow as tf


@tf.function
def change_partition_by_name(in_partition, in_partition_type: str, out_partition_type: str):
    """Switch between partition types. Only for 1-D partition tensors. Uses RaggedTensor partition scheme naming.
    Not all partition schemes are fully complete or can be converted into each other.

    Args:
        in_partition (tf.Tensor): Row partition tensor of shape (N, ).
        in_partition_type (str): Source partition type, can be either 'row_splits', 'row_length', 'value_rowids',
            'row_starts' or 'row_limits'
        out_partition_type (str): Target partition type, can be either 'row_splits', 'row_length', 'value_rowids',
            'row_starts' or 'row_limits'

    Returns:
        tf.Tensor: Row partition tensor of target type.
    """
    if in_partition_type == out_partition_type:
        # Do nothing here
        out_partition = in_partition

    # row_lengths
    elif in_partition_type in ["row_length", "row_lengths"] and out_partition_type in ["row_split", "row_splits"]:
        # We need ex. (1,2,3) -> (0,1,3,6)
        out_partition = tf.pad(tf.cumsum(in_partition), [[1, 0]])

    elif in_partition_type in ["row_length", "row_lengths"] and out_partition_type == "value_rowids":
        out_partition = tf.repeat(tf.range(tf.shape(in_partition)[0]), in_partition)

    elif in_partition_type in ["row_length", "row_lengths"] and out_partition_type in ["row_start", "row_starts"]:
        out_partition = tf.cumsum(in_partition, exclusive=True)

    elif in_partition_type in ["row_length", "row_lengths"] and out_partition_type in ["row_limit", "row_limits"]:
        out_partition = tf.cumsum(in_partition)

    # row_splits
    elif in_partition_type in ["row_split", "row_splits"] and out_partition_type in ["row_length", "row_lengths"]:
        # Matches length if (0,1,3,6) -> (1,2,3)
        out_partition = in_partition[1:] - in_partition[:-1]

    elif in_partition_type in ["row_split", "row_splits"] and out_partition_type == "value_rowids":
        # Get row_length
        part_sum = in_partition[1:] - in_partition[:-1]
        out_partition = tf.repeat(tf.range(tf.shape(part_sum)[0]), part_sum)

    elif in_partition_type in ["row_split", "row_splits"] and out_partition_type in ["row_limit", "row_limits"]:
        out_partition = in_partition[1:]

    elif in_partition_type in ["row_split", "row_splits"] and out_partition_type in ["row_start", "row_starts"]:
        out_partition = in_partition[:-1]

    # value_rowids
    elif in_partition_type == "value_rowids" and out_partition_type in ["row_length", "row_lengths"]:
        out_partition = tf.math.segment_sum(tf.ones_like(in_partition), in_partition)

    elif in_partition_type == "value_rowids" and out_partition_type in ["row_split", "row_splits"]:
        # Get row_length
        part_sum = tf.math.segment_sum(tf.ones_like(in_partition), in_partition)
        out_partition = tf.pad(tf.cumsum(part_sum), [[1, 0]])

    elif in_partition_type == "value_rowids" and out_partition_type in ["row_limit", "row_limits"]:
        part_sum = tf.math.segment_sum(tf.ones_like(in_partition), in_partition)
        out_partition = tf.cumsum(part_sum)

    elif in_partition_type == "value_rowids" and out_partition_type in ["row_start", "row_starts"]:
        part_sum = tf.math.segment_sum(tf.ones_like(in_partition), in_partition)
        out_partition = tf.cumsum(part_sum, exclusive=True)

    # row_starts
    elif in_partition_type in ["row_start", "row_starts"]:
        raise ValueError("Can not infer partition scheme from row_starts alone, missing nvals")

    # row_starts
    elif in_partition_type in ["row_limit", "row_limits"] and out_partition_type in ["row_length", "row_lengths"]:
        part_split = tf.pad(in_partition, [[1, 0]])
        out_partition = part_split[1:] - part_split[:-1]

    elif in_partition_type in ["row_limit", "row_limits"] and out_partition_type == "value_rowids":
        part_split = tf.pad(in_partition, [[1, 0]])
        part_sum = part_split[1:] - part_split[:-1]
        out_partition = tf.repeat(tf.range(tf.shape(part_sum)[0]), part_sum)

    elif in_partition_type in ["row_limit", "row_limits"] and out_partition_type in ["row_split", "row_splits"]:
        out_partition = tf.pad(in_partition, [[1, 0]])

    elif in_partition_type in ["row_limit", "row_limits"] and out_partition_type in ["row_start", "row_starts"]:
        out_partition = tf.pad(in_partition, [[1, 0]])[:-1]

    else:
        raise TypeError("Unknown partition scheme, use: 'value_rowids', 'row_splits', 'row_lengths', etc.")

    return out_partition


@tf.function
def partition_row_indexing(tensor_index, part_target, part_index,
                           partition_type_target,
                           partition_type_index,
                           from_indexing: str = 'sample',
                           to_indexing: str = 'batch'):
    """Change the indices in an index-tensor to reference within row partition. Assuming the indices refer to the rows
    of a values tensor, introducing a row-partition for value and index tensor requires the indices to be shifted
    accordingly. Provided the first dimension is the batch-dimension, the change of indices would then change from
    a sample to a batch assignment or vice versa. For graph networks this can be between e.g. edge-indices referring to
    nodes in each graph or in case of batch-assignment, this is equivalent to the so-called disjoint representation.
    To change indices, the row partition of index and target tensor must be known.

    .. code-block:: python

        import tensorflow as tf
        values = tf.RaggedTensor.from_row_lengths([10, 20, 30, 40], [2, 2])
        indices = tf.RaggedTensor.from_row_lengths([0, 0, 1, 1], [3, 1])
        print(tf.gather(values, indices, batch_dims=1))
        # <tf.RaggedTensor [[10, 10, 20], [40]]>
        indices_batch = partition_row_indexing(tf.constant([0, 0, 1, 1]), tf.constant([2, 2]),
            tf.constant([3, 2]), 'row_lengths', 'row_lengths')
        # <tf.Tensor: shape=(4,), dtype=int32, numpy=array([0, 0, 1, 3])>
        print(tf.gather(tf.constant([10, 20, 30, 40]), indices_batch))
        # <tf.Tensor: shape=(4,), dtype=int32, numpy=array([10, 10, 20, 40])>
        # Same result as gather with batch_dims=1 from above

    Args:
        tensor_index (tf.Tensor): Tensor containing indices for row-values of shape `(None, ...)`.
        part_target (tf.Tensor): Partition tensor of the values tensor. The value tensor itself is not required.
        part_index (tf.Tensor): Partition tensor of the index (input) tensor.
        partition_type_target (str:) Partition scheme of target value tensor, e.g. 'row_splits', etc.
        partition_type_index (str): Partition scheme of index tensor, e.g. 'row_splits', etc.
        from_indexing (str): Indexing reference of the input. Can either be 'sample' or 'batch'.
        to_indexing (str): Indexing reference of the output. Can either be 'sample' or 'batch'.

    Returns:
        tf.Tensor: Index tensor with shifted indices.
    """
    if to_indexing == from_indexing:
        # Nothing to do here
        return tensor_index

    # we need node row_splits
    nod_splits = change_partition_by_name(part_target, partition_type_target, "row_splits")

    # we need edge value_rowids
    edge_ids = change_partition_by_name(part_index, partition_type_index, "value_rowids")

    # Just gather the splits i.e. index offset for each graph id
    shift_index = tf.gather(nod_splits, edge_ids)

    # Expand dimension to broadcast to indices for suitable axis
    # The shift_index is always 1D tensor. Add further (N, 1, 1, ...)
    for i in range(1, tensor_index.shape.rank):
        shift_index = tf.expand_dims(shift_index, axis=-1)

    # Add or remove batch offset from index tensor
    if to_indexing == 'batch' and from_indexing == 'sample':
        indexlist = tensor_index + tf.cast(shift_index, dtype=tensor_index.dtype)
    elif to_indexing == 'sample' and from_indexing == 'batch':
        indexlist = tensor_index - tf.cast(shift_index, dtype=tensor_index.dtype)
    else:
        raise TypeError("ERROR:kgcnn: Unknown index change, use: 'sample', 'batch', ...")

    out = indexlist
    return out

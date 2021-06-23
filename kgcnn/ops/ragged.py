import tensorflow as tf


@tf.function
def ragged_tensor_from_partition_by_name(value, part, partition_type, ragged_validate):
    # Not for nested ragged definition
    if partition_type in ["row_length", "row_lengths"]:
        out = tf.RaggedTensor.from_row_lengths(value, part, validate=ragged_validate)
    elif partition_type in ["row_split", "row_splits"]:
        out = tf.RaggedTensor.from_row_splits(value, part, validate=ragged_validate)
    elif partition_type == "value_rowids":
        out = tf.RaggedTensor.from_value_rowids(value, part, validate=ragged_validate)
    elif partition_type in ["row_limit", "row_limits"]:
        out = tf.RaggedTensor.from_row_limits(value, part, validate=ragged_validate)
    elif partition_type in ["row_start", "row_starts"]:
        out = tf.RaggedTensor.from_row_starts(value, part, validate=ragged_validate)
    else:
        raise TypeError("Unknown partition scheme, use: 'row_lengths', 'row_splits', ...")
    return out


@tf.function
def partition_from_ragged_tensor_by_name(tens, partition_type):
    # Not for nested ragged definition
    flat_tens = tens.values
    if partition_type in ["row_length", "row_lengths"]:
        outpart = tens.row_lengths()
    elif partition_type in ["row_split", "row_splits"]:
        outpart = tens.row_splits
    elif partition_type == "value_rowids":
        outpart = tens.value_rowids()
    elif partition_type in ["row_limit", "row_limits"]:
        outpart = tens.row_limits()
    elif partition_type in ["row_start", "row_starts"]:
        outpart = tens.row_starts()
    else:
        raise TypeError("Unknown partition scheme, use: 'row_lengths', 'row_splits', ...")

    return [flat_tens, outpart]
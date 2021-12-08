import tensorflow as tf


@tf.function
def ragged_tensor_from_partition_by_name(value, part, partition_type, ragged_validate=False, name=None):
    """Construct ragged tensor from value tensor plus partition `tf.Tensor`. The type of the row partition can be passed
    as string identifier.

    Args:
        value (tf.Tensor): A potentially ragged tensor with shape `[nvals, ...]`.
        part (tf.Tensor): A 1-D integer tensor with partition info. The partition tensor must adhere the partition rules
            for the corresponding `tf.ragged` methods.
        partition_type (str): String identifier of the partition scheme. Either `row_splits`, `row_limits` etc.
        ragged_validate (bool): If true, then use assertions to check that the arguments form a valid RaggedTensor.
            Note: these assertions incur a runtime cost, since they must be checked for each tensor value.
        name (str): A name prefix for the RaggedTensor (optional).

    Returns:
        tf.RaggedTensor: Creates a RaggedTensor with rows partitioned by partition tensor.
    """
    # Not for nested ragged definition
    if partition_type in ["row_length", "row_lengths"]:
        out = tf.RaggedTensor.from_row_lengths(value, part, validate=ragged_validate, name=name)
    elif partition_type in ["row_split", "row_splits"]:
        out = tf.RaggedTensor.from_row_splits(value, part, validate=ragged_validate, name=name)
    elif partition_type == "value_rowids":
        out = tf.RaggedTensor.from_value_rowids(value, part, validate=ragged_validate, name=name)
    elif partition_type in ["row_limit", "row_limits"]:
        out = tf.RaggedTensor.from_row_limits(value, part, validate=ragged_validate, name=name)
    elif partition_type in ["row_start", "row_starts"]:
        out = tf.RaggedTensor.from_row_starts(value, part, validate=ragged_validate, name=name)
    else:
        raise TypeError("Unknown partition scheme, use: 'row_lengths', 'row_splits', ...")
    return out


@tf.function
def partition_from_ragged_tensor_by_name(ragged_tensor, partition_type: str):
    """Extract row partition tensor from ragged tensor defined by string identifier of the partition scheme.

    Args:
        ragged_tensor (tf.RaggedTensor): Ragged tensor to extract row partition from.
        partition_type (str): String identifier of the partition scheme. Either `row_splits`, `row_limits` etc.

    Returns:
        tf.Tensor: Row partition defined by partition-type.
    """
    # Not for nested ragged definition
    flat_tens = ragged_tensor.values
    if partition_type in ["row_length", "row_lengths"]:
        out_part = ragged_tensor.row_lengths()
    elif partition_type in ["row_split", "row_splits"]:
        out_part = ragged_tensor.row_splits
    elif partition_type == "value_rowids":
        out_part = ragged_tensor.value_rowids()
    elif partition_type in ["row_limit", "row_limits"]:
        out_part = ragged_tensor.row_limits()
    elif partition_type in ["row_start", "row_starts"]:
        out_part = ragged_tensor.row_starts()
    else:
        raise TypeError("Unknown partition scheme, use: 'row_lengths', 'row_splits', ...")

    return [flat_tens, out_part]
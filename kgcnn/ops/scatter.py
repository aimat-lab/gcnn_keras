import tensorflow as tf
# import tensorflow.keras as ks


@tf.function
def tensor_scatter_nd_mean(tensor, indices, updates, name=None):
    """Temporary replacement of tensor_scatter_nd_mean until its supported by tensorflow.

    Args:
        tensor (tf.Tensor): Tensor to scatter updates into.
        indices (tf.Tensor): Indices to for updates.
        updates (tf.Tensor): Updates of new entries for tensor.
        name (str): Name of the tensor.

    Returns:
        tf.Tensor: Updates scattered into tensor with mean update
    """
    pass


@tf.function
def tensor_scatter_nd_ops_by_name(segment_name, tensor, indices, updates, name=None):
    """Scatter operation chosen by name that can replace segment-operations.

    Args:
        segment_name (str): Operation to update scattered updates. Either 'sum' or 'min' etc.
        tensor (tf.Tensor): Tensor to scatter updates into.
        indices (tf.Tensor): Indices to for updates.
        updates (tf.Tensor): Updates of new entries for tensor.
        name (str): Name of the tensor.

    Returns:
        tf.Tensor: Updates scattered into tensor with different update rules.
    """
    if segment_name in ["segment_mean", "mean", "reduce_mean"]:
        pool = tensor_scatter_nd_mean(tensor, indices, updates, name=name)
    elif segment_name in ["segment_sum", "sum", "reduce_sum"]:
        pool = tf.tensor_scatter_nd_add(tensor, indices, updates, name=name)
    elif segment_name in ["segment_max", "max", "reduce_max"]:
        pool = tf.tensor_scatter_nd_max(tensor, indices, updates, name=name)
    elif segment_name in ["segment_min", "sum", "reduce_min"]:
        pool = tf.tensor_scatter_nd_min(tensor, indices, updates, name=name)
    else:
        raise TypeError("Unknown pooling, choose: 'mean', 'sum', ...")
    return pool

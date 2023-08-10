import tensorflow as tf


def tensor_scatter_nd_mean(tensor, indices, updates, name=None):
    """Tensor scatter with mean updates.

    Args:
        tensor (tf.Tensor): Tensor to scatter updates into.
        indices (tf.Tensor): Indices to for updates.
        updates (tf.Tensor): Updates of new entries for tensor.
        name (str): Name of the tensor.

    Returns:
        tf.Tensor: Updates scattered into tensor with mean update.
    """
    # Simply placeholder for `tensor_scatter_nd_mean` .
    # Not very efficient for sparse updates. However, for simple aggregation, where each entry gets at least one
    # update, this should not be too bad.
    values_added = tf.tensor_scatter_nd_add(tensor, indices, updates, name=name)
    values_count = tf.tensor_scatter_nd_add(tf.ones_like(tensor), indices, tf.ones_like(updates))
    return values_added/values_count


@tf.function
def tensor_scatter_nd_ops_by_name(scatter_name, tensor, indices, updates, name=None):
    """Scatter operation chosen by name that pick tensor_scatter_nd functions.

    Args:
        scatter_name (str): Operation to update scattered updates. Either 'sum' or 'min' etc.
        tensor (tf.Tensor): Tensor to scatter updates into.
        indices (tf.Tensor): Indices to for updates.
        updates (tf.Tensor): Updates of new entries for tensor.
        name (str): Name of the tensor.

    Returns:
        tf.Tensor: Updates scattered into tensor with different update rules.
    """
    if scatter_name in ["segment_sum", "sum", "reduce_sum", "add", "scatter_add"]:
        pool = tf.tensor_scatter_nd_add(tensor, indices, updates, name=name)
    elif scatter_name in ["segment_max", "max", "reduce_max", "scatter_max"]:
        pool = tf.tensor_scatter_nd_max(tensor, indices, updates, name=name)
    elif scatter_name in ["segment_min", "min", "reduce_min", "scatter_min"]:
        pool = tf.tensor_scatter_nd_min(tensor, indices, updates, name=name)
    elif scatter_name in ["segment_mean", "mean", "reduce_mean", "scatter_mean"]:
        pool = tensor_scatter_nd_mean(tensor, indices, updates, name=name)
    else:
        raise TypeError("Unknown pooling, choose: 'mean', 'sum', ...")
    return pool

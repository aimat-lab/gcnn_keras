import tensorflow as tf
import tensorflow.keras as ks


@tf.function
def scatter_nd_segment(data, segment_ids, target_shape):
    """Scatter output of segment operation into target shape.
    This additional step is required since segment_ids may not contain largest id but target_shape must
    match with largest id.

    Args:
        data (tf.Tensor): Output of segment operation.
        segment_ids (tf.Tensor): Former ids of segment operation. 1D Tensor.
        target_shape (tf.TensorShape): of target tensor to scatter output into.

    Returns:
        tf.Tensor: tensor with scattered data
    """
    # If max id is not in segment_ids requires scatter into tensor of correct shape[0] that matches target first dim
    # out_target_shape = (tf.shape(nod, out_type=nodind.dtype)[0], ks.backend.int_shape(dens)[-1])
    out_target_shape = tf.concat([target_shape[:1], tf.shape(data)[1:]], axis=0)
    # Make segment indices
    segment_index = tf.range(tf.shape(data)[0])
    out_tensor = tf.scatter_nd(ks.backend.expand_dims(segment_index, axis=-1), data, out_target_shape)
    return out_tensor


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
    pool = None
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


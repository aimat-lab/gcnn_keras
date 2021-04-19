import tensorflow as tf
import tensorflow.keras as ks





@tf.function
def _scatter_segment_tensor_nd(data, segment_ids, target_shape):
    """Scatter output of segment operation into target shape.

    Args:
        data (tf.tensor): Output of segment operation.
        segment_ids (tf.tensor): Former ids of segment operation. 1D Tensor.
        target_shape: tf.shape of target tensor to scatter output into.

    Returns:

    """
    # If max id is not in segment_ids requires scatter into tensor of correct shape[0] that matches target firs dim
    # out_target_shape = (tf.shape(nod, out_type=nodind.dtype)[0], ks.backend.int_shape(dens)[-1])
    out_target_shape = tf.concat([target_shape[:1] ,tf.shape(data)[1:]],axis=0)
    # Make segment indices
    segment_index = tf.range(tf.shape(data)[0])
    out_tensor = tf.scatter_nd(ks.backend.expand_dims(segment_index, axis=-1), data, out_target_shape)

    return out_tensor


@tf.function
def _tensor_scatter_nd_mean(tensor, indices, updates):
    out_values = tf.tensor_scatter_nd_add(tensor, indices, updates)
    # If updates are slices, don't need to build full rank tensor to count updates
    counts_tensor = tf.zeros(tf.concat([tf.shape(tensor)[:tf.shape(indices)[-1]],tf.ones_like(tf.shape(tensor))[tf.shape(indices)[-1]:]],axis=0),dtype=tensor.dtype)
    counts_updates = tf.ones(tf.concat([tf.shape(updates)[:tf.shape(indices)[-1]],tf.ones_like(tf.shape(updates))[tf.shape(indices)[-1]:]],axis=0),dtype=updates.dtype)
    num_values = tf.tensor_scatter_nd_add(counts_tensor, indices, counts_updates)
    out = tf.math.divide_no_nan(out_values,num_values)
    return out

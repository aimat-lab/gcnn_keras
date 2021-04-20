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
    # If max id is not in segment_ids requires scatter into tensor of correct shape[0] that matches target first dim
    # out_target_shape = (tf.shape(nod, out_type=nodind.dtype)[0], ks.backend.int_shape(dens)[-1])
    out_target_shape = tf.concat([target_shape[:1], tf.shape(data)[1:]], axis=0)
    # Make segment indices
    segment_index = tf.range(tf.shape(data)[0])
    out_tensor = tf.scatter_nd(ks.backend.expand_dims(segment_index, axis=-1), data, out_target_shape)

    return out_tensor



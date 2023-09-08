import tensorflow as tf


def scatter_reduce_min(indices, values, shape):
    target = tf.fill(shape, values.dtype.limits[1], dtype=values.dtype)
    return tf.tensor_scatter_nd_min(target, indices, values)


def scatter_reduce_max(indices, values, shape):
    target = tf.fill(shape, values.dtype.limits[0], dtype=values.dtype)
    return tf.tensor_scatter_nd_max(target, indices, values)


def scatter_reduce_mean(indices, values, shape):
    counts = tf.scatter_nd(indices, tf.ones_like(values), shape)
    return tf.scatter_nd(indices, values, shape)/counts


def scatter_reduce_sum(indices, values, shape):
    return tf.scatter_nd(indices, values, tf.cast(shape, dtype="int64"))


def scatter_reduce_softmax(indices, values, shape):
    # if normalize:
    #     data_segment_max = tf.math.segment_max(data, segment_ids)
    #     data_max = tf.gather(data_segment_max, segment_ids)
    #     data = data - data_max

    values_exp = tf.math.exp(values)
    values_exp_sum = tf.scatter_nd(indices, values_exp, tf.cast(shape, dtype="int64"))
    values_exp_sum = tf.gather(values_exp_sum, indices, axis=0)
    return values_exp / values_exp_sum

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
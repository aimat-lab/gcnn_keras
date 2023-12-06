import tensorflow as tf


def scatter_reduce_sum(indices, values, shape):
    indices = tf.expand_dims(indices, axis=1)
    return tf.scatter_nd(indices, values, tf.cast(shape, dtype="int64"))


def scatter_reduce_min(indices, values, shape):
    indices = tf.expand_dims(indices, axis=1)
    target = tf.cast(tf.fill(shape, values.dtype.max), dtype=values.dtype)
    return tf.tensor_scatter_nd_min(target, indices, values)


def scatter_reduce_max(indices, values, shape):
    indices = tf.expand_dims(indices, axis=1)
    target = tf.cast(tf.fill(shape, values.dtype.min), dtype=values.dtype)
    return tf.tensor_scatter_nd_max(target, indices, values)


def scatter_reduce_mean(indices, values, shape):
    indices = tf.expand_dims(indices, axis=1)
    counts = tf.scatter_nd(indices, tf.ones_like(values), shape)
    return tf.math.divide_no_nan(tf.scatter_nd(indices, values, shape), counts)


def scatter_reduce_softmax(indices, values, shape, normalize: bool = False):
    indices_scatter = tf.expand_dims(indices, axis=1)

    if normalize:
        zeros_min = tf.zeros(shape, dtype=values.dtype)  # Zero is okay here
        data_segment_max = tf.tensor_scatter_nd_max(zeros_min, indices_scatter, values)
        data_max = tf.gather(data_segment_max, indices, axis=0)
        values = values - data_max

    values_exp = tf.math.exp(values)
    values_exp_sum = tf.scatter_nd(indices_scatter, values_exp, tf.cast(shape, dtype="int64"))
    values_exp_sum = tf.gather(values_exp_sum, indices, axis=0)
    return values_exp / values_exp_sum


def repeat_static_length(x, repeats, axis=None, total_repeat_length: int = None):
    return tf.repeat(x, repeats=repeats, axis=axis)


def decompose_ragged_tensor(x, batch_dtype="int64"):
    row_ids = tf.cast(x.value_rowids(), dtype=batch_dtype)
    row_lengths = tf.cast(x.row_lengths(), dtype=batch_dtype)
    row_starts = tf.cast(x.row_starts(), dtype=batch_dtype)
    in_ids = tf.range(tf.shape(row_ids)[0], dtype=batch_dtype)
    in_ids = in_ids - tf.repeat(row_starts, row_lengths, axis=0)
    # in_ids = tf.ragged.range(row_lengths).values
    return x.values, row_ids, in_ids, row_lengths


def norm(x, ord='fro', axis=None, keepdims=False):
    return tf.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def cross(x1, x2):
    return tf.linalg.cross(x1, x2)













import tensorflow as tf


def scatter_min(indices, values, shape):
    return tf.scatter_min(indices, values, shape)

def scatter_max(indices, values, shape):
    return tf.scatter_nd(indices, values, shape)

def scatter_mean(indices, values, shape):

    return tf.scatter_nd(indices, values, shape)

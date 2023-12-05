import jax.numpy as jnp


def scatter_reduce_sum(indices, values, shape):
    zeros = jnp.zeros(shape, values.dtype)
    return zeros.at[indices].add(values)


def scatter_reduce_min(indices, values, shape):
    zeros = jnp.zeros(shape, values.dtype.max, values.dtype)
    return zeros.at[indices].min(values)


def scatter_reduce_max(indices, values, shape):
    zeros = jnp.full(shape, values.dtype.min, values.dtype)
    return zeros.at[indices].max(values)


def scatter_reduce_mean(indices, values, shape):
    zeros = jnp.zeros(shape, values.dtype)
    counts = jnp.zeros(shape, values.dtype)
    counts = counts.at[indices].add(jnp.ones_like(values))
    return zeros.at[indices].add(values)/counts


def scatter_reduce_softmax(indices, values, shape, normalize: bool = False):

    if normalize:
        zeros_min = jnp.zeros(shape, values.dtype)  # Zero is okay here
        data_segment_max = zeros_min.at[indices].max(values)
        data_max = jnp.take(data_segment_max, indices, axis=0)
        values = values - data_max

    values_exp = jnp.exp(values)
    zeros = jnp.zeros(shape, values.dtype)
    values_exp_sum = zeros.at[indices].add(values_exp)
    values_exp_sum = jnp.take(values_exp_sum, indices, axis=0)
    return values_exp / values_exp_sum


def repeat_static_length(x, repeats, axis=None, total_repeat_length: int = None):
    return jnp.repeat(x, repeats=repeats, axis=axis, total_repeat_length=total_repeat_length)


def decompose_ragged_tensor(x):
    raise NotImplementedError("Operation supported this backend '%s'." % __name__)


def norm(x, ord='fro', axis=None, keepdims=False):
    return jnp.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def cross(x1, x2):
    return jnp.cross(x1, x2, axis=-1)

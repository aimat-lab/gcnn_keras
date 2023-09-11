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
    counts.at[indices].add(jnp.ones_like(values))
    return zeros.at[indices].add(values)/counts

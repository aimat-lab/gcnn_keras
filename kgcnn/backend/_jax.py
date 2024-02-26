import numpy as np
import jax.numpy as jnp
from kgcnn import __safe_scatter_max_min_to_zero__ as global_safe_scatter_max_min_to_zero


class binfo:
  kind: str = "b"
  bits: int = 8  # May be different for jax.
  min: bool = False
  max: bool = True
  dtype: np.dtype = "bool"


def dtype_infos(dtype):
    if dtype.kind in ["f", "c"]:
        return jnp.finfo(dtype)
    elif dtype.kind in ["i", "u"]:
        return jnp.iinfo(dtype)
    elif dtype.kind in ["b"]:
        return binfo()
    else:
        raise TypeError("Unknown dtype '%s' to get type info." % dtype)


def scatter_reduce_sum(indices, values, shape):
    zeros = jnp.zeros(shape, values.dtype)
    return zeros.at[indices].add(values)


def scatter_reduce_min(indices, values, shape):
    max_of_dtype = dtype_infos(values.dtype).max
    zeros = jnp.full(shape, max_of_dtype, values.dtype)
    out = zeros.at[indices].min(values)
    if global_safe_scatter_max_min_to_zero:
        has_scattered = jnp.zeros(shape, "bool")
        has_scattered = has_scattered.at[indices].set(jnp.ones_like(values, dtype="bool"))
        out = jnp.where(has_scattered, out, jnp.zeros_like(out))
    return out


def scatter_reduce_max(indices, values, shape):
    min_of_dtype = dtype_infos(values.dtype).min
    zeros = jnp.full(shape, min_of_dtype, values.dtype)
    out = zeros.at[indices].max(values)
    if global_safe_scatter_max_min_to_zero:
        has_scattered = jnp.zeros(shape, "bool")
        has_scattered = has_scattered.at[indices].set(jnp.ones_like(values, dtype="bool"))
        out = jnp.where(has_scattered, out, jnp.zeros_like(out))
    return out


def scatter_reduce_mean(indices, values, shape):
    zeros = jnp.zeros(shape, values.dtype)
    counts = jnp.zeros(shape, values.dtype)
    counts = counts.at[indices].add(jnp.ones_like(values))
    inverse_counts = jnp.nan_to_num(jnp.reciprocal(counts), posinf=0.0, neginf=0.0, nan=0.0)
    return zeros.at[indices].add(values)*inverse_counts


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
    raise NotImplementedError("Operation not supported by this backend '%s'." % __name__)


def norm(x, ord='fro', axis=None, keepdims=False):
    return jnp.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def cross(x1, x2):
    return jnp.cross(x1, x2, axis=-1)

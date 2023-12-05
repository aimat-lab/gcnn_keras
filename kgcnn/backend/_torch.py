import torch


def scatter_reduce_sum(indices, values, shape):
    dims_to_add = values.dim() - indices.dim()
    for _ in range(dims_to_add):
        indices = torch.unsqueeze(indices, dim=-1)
    return torch.zeros(*shape, dtype=values.dtype, device=values.device).scatter_reduce(
        0, torch.broadcast_to(indices, values.shape), values, reduce='sum')


def scatter_reduce_min(indices, values, shape):
    dims_to_add = values.dim() - indices.dim()
    for _ in range(dims_to_add):
        indices = torch.unsqueeze(indices, dim=-1)
    return torch.zeros(*shape, dtype=values.dtype, device=values.device).scatter_reduce(
        0, torch.broadcast_to(indices, values.shape), values, reduce='amin', include_self=False)


def scatter_reduce_max(indices, values, shape):
    dims_to_add = values.dim() - indices.dim()
    for _ in range(dims_to_add):
        indices = torch.unsqueeze(indices, dim=-1)
    return torch.zeros(*shape, dtype=values.dtype, device=values.device).scatter_reduce(
        0, torch.broadcast_to(indices, values.shape), values, reduce='amax', include_self=False)


def scatter_reduce_mean(indices, values, shape):
    dims_to_add = values.dim() - indices.dim()
    for _ in range(dims_to_add):
        indices = torch.unsqueeze(indices, dim=-1)
    return torch.zeros(*shape, dtype=values.dtype, device=values.device).scatter_reduce(
        0, torch.broadcast_to(indices, values.shape), values, reduce='mean', include_self=False)


def scatter_reduce_softmax(indices, values, shape, normalize: bool = False):
    indices_scatter = indices
    dims_to_add = values.dim()-indices.dim()
    for _ in range(dims_to_add):
        indices_scatter = torch.unsqueeze(indices_scatter, dim=-1)

    if normalize:
        zeros_min = torch.zeros(*shape, dtype=values.dtype, device=values.device)
        data_segment_max = zeros_min.scatter_reduce(
            0, torch.broadcast_to(indices_scatter, values.shape), values, reduce='amax', include_self=False)
        data_max = torch.index_select(data_segment_max, dim=0, index=indices)
        values = values - data_max

    values_exp = torch.exp(values)
    zeros = torch.zeros(*shape, dtype=values.dtype, device=values.device)
    values_exp_sum = zeros.scatter_reduce(
        0, torch.broadcast_to(indices_scatter, values_exp.shape), values_exp, reduce='sum', include_self=True)
    values_exp_sum = torch.index_select(values_exp_sum, dim=0, index=indices)
    return values_exp / values_exp_sum


def repeat_static_length(x, repeats, axis=None, total_repeat_length: int = None):
    # from keras_core.backend.torch.numpy import repeat
    return torch.repeat_interleave(x, repeats, dim=axis)


def decompose_ragged_tensor(x):
    raise NotImplementedError("Operation supported this backend '%s'." % __name__)


def norm(x, ord='fro', axis=None, keepdims=False):
    return torch.linalg.norm(x, ord=ord, dim=axis, keepdims=keepdims)


def cross(x1, x2):
    return torch.cross(x1, x2, dim=-1)

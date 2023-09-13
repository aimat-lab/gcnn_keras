import torch


def scatter_reduce_sum(indices, values, shape):
    indices = torch.unsqueeze(indices, dim=-1)
    return torch.zeros(*shape, dtype=values.dtype, device=values.device).scatter_reduce(
        0, torch.broadcast_to(indices, values.shape), values, reduce='sum')


def scatter_reduce_min(indices, values, shape):
    indices = torch.unsqueeze(indices, dim=-1)
    return torch.zeros(*shape, dtype=values.dtype, device=values.device).scatter_reduce(
        0, torch.broadcast_to(indices, values.shape), values, reduce='amin', include_self=False)


def scatter_reduce_max(indices, values, shape):
    indices = torch.unsqueeze(indices, dim=-1)
    return torch.zeros(*shape, dtype=values.dtype, device=values.device).scatter_reduce(
        0, torch.broadcast_to(indices, values.shape), values, reduce='amax', include_self=False)


def scatter_reduce_mean(indices, values, shape):
    indices = torch.unsqueeze(indices, dim=-1)
    return torch.zeros(*shape, dtype=values.dtype, device=values.device).scatter_reduce(
        0, torch.broadcast_to(indices, values.shape), values, reduce='mean', include_self=False)


def scatter_reduce_softmax(indices, values, shape, normalize: bool = False):
    indices_scatter = torch.unsqueeze(indices, dim=-1)

    if normalize:
        zeros_min = torch.zeros(*shape, values.dtype.limits[0], dtype=values.dtype)
        data_segment_max = zeros_min.scatter_reduce(
            0, torch.broadcast_to(indices_scatter, values.shape), values, reduce='amax', include_self=False)
        data_max = data_segment_max[indices]
        values = values - data_max

    values_exp = torch.exp(values)
    values_exp_sum = torch.zeros(*shape, values.dtype, device=values.device)
    values_exp_sum.scatter_reduce(0, torch.broadcast_to(indices_scatter, values.shape), values_exp, reduce='sum')
    values_exp_sum = values_exp_sum[indices]
    return values_exp / values_exp_sum


def repeat_static_length(x, repeats, axis=None, total_repeat_length: int = None):
    # from keras_core.backend.torch.numpy import repeat
    return torch.repeat_interleave(x, repeats, dim=axis)

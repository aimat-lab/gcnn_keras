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


def scatter_reduce_softmax(indices, values, shape):
    indices = torch.unsqueeze(indices, dim=-1)
    return torch.zeros(*shape, dtype=values.dtype, device=values.device).scatter_reduce(
        0, torch.broadcast_to(indices, values.shape), values, reduce='sum')

import torch
from keras_core import ops


def scatter_sum(indices, values, shape):
    return torch.zeros(*shape, dtype=values.dtype).scatter_reduce(
        0, torch.broadcast_to(indices, values.shape), values, reduce='sum')


def scatter_min(indices, values, shape):
    return torch.zeros(*shape, dtype=values.dtype).scatter_reduce(
        0, torch.broadcast_to(indices, values.shape), values, reduce='amin', include_self=False)


def scatter_max(indices, values, shape):
    return torch.zeros(*shape, dtype=values.dtype).scatter_reduce(
        0, torch.broadcast_to(indices, values.shape), values, reduce='amax', include_self=False)


def scatter_mean(indices, values, shape):
    return torch.zeros(*shape, dtype=values.dtype).scatter_reduce(
        0, torch.broadcast_to(indices, values.shape), values, reduce='mean', include_self=False)

import kgcnn.backend as kgcnn_backend
from keras import KerasTensor
from kgcnn.backend import any_symbolic_tensors
from keras import Operation


class _ScatterMax(Operation):
    def call(self, indices, values, shape):
        return kgcnn_backend.scatter_reduce_max(indices, values, shape)

    def compute_output_spec(self, indices, values, shape):
        return KerasTensor(shape, dtype=values.dtype)


def scatter_reduce_max(indices, values, shape):
    r"""Scatter values at indices into new tensor of shape.

    Args:
        indices (Tensor): 1D Indices of shape `(M, )` .
        values (Tensor): Vales of shape `(M, ...)` .
        shape (tuple): Target shape.

    Returns:
        Tensor: Scattered values of `shape` .
    """
    if any_symbolic_tensors((indices, values, shape)):
        return _ScatterMax().symbolic_call(indices, values, shape)
    return kgcnn_backend.scatter_reduce_max(indices, values, shape)


class _ScatterMin(Operation):
    def call(self, indices, values, shape):
        return kgcnn_backend.scatter_reduce_min(indices, values, shape)

    def compute_output_spec(self, indices, values, shape):
        return KerasTensor(shape, dtype=values.dtype)


def scatter_reduce_min(indices, values, shape):
    r"""Scatter values at indices into new tensor of shape.

    Args:
        indices (Tensor): 1D Indices of shape `(M, )` .
        values (Tensor): Vales of shape `(M, ...)` .
        shape (tuple): Target shape.

    Returns:
        Tensor: Scattered values of `shape` .
    """
    if any_symbolic_tensors((indices, values, shape)):
        return _ScatterMin().symbolic_call(indices, values, shape)
    return kgcnn_backend.scatter_reduce_min(indices, values, shape)


class _ScatterMean(Operation):
    def call(self, indices, values, shape):
        return kgcnn_backend.scatter_reduce_mean(indices, values, shape)

    def compute_output_spec(self, indices, values, shape):
        return KerasTensor(shape, dtype=values.dtype)


def scatter_reduce_mean(indices, values, shape):
    r"""Scatter values at indices into new tensor of shape.

    Args:
        indices (Tensor): 1D Indices of shape `(M, )` .
        values (Tensor): Vales of shape `(M, ...)` .
        shape (tuple): Target shape.

    Returns:
        Tensor: Scattered values of `shape` .
    """
    if any_symbolic_tensors((indices, values, shape)):
        return _ScatterMean().symbolic_call(indices, values, shape)
    return kgcnn_backend.scatter_reduce_mean(indices, values, shape)


class _ScatterSum(Operation):
    def call(self, indices, values, shape):
        return kgcnn_backend.scatter_reduce_sum(indices, values, shape)

    def compute_output_spec(self, indices, values, shape):
        return KerasTensor(shape, dtype=values.dtype)


def scatter_reduce_sum(indices, values, shape):
    r"""Scatter values at indices into new tensor of shape.

    Args:
        indices (Tensor): 1D Indices of shape `(M, )` .
        values (Tensor): Vales of shape `(M, ...)` .
        shape (tuple): Target shape.

    Returns:
        Tensor: Scattered values of `shape` .
    """
    if any_symbolic_tensors((indices, values, shape)):
        return _ScatterSum().symbolic_call(indices, values, shape)
    return kgcnn_backend.scatter_reduce_sum(indices, values, shape)


class _ScatterSoftmax(Operation):

    def __init__(self, normalize: bool = False):
        super().__init__()
        self.normalize = normalize

    def call(self, indices, values, shape):
        return kgcnn_backend.scatter_reduce_softmax(indices, values, shape, normalize=self.normalize)

    def compute_output_spec(self, indices, values, shape):
        return KerasTensor(shape, dtype=values.dtype)


def scatter_reduce_softmax(indices, values, shape, normalize: bool = False):
    r"""Scatter values at indices to normalize values via softmax.

    Args:
        indices (Tensor): 1D Indices of shape `(M, )` .
        values (Tensor): Vales of shape `(M, ...)` .
        shape (tuple): Target shape of scattered tensor.

    Returns:
        Tensor: Values with softmax computed by grouping at indices.
    """
    if any_symbolic_tensors((indices, values, shape)):
        return _ScatterSoftmax(normalize=normalize).symbolic_call(indices, values, shape)
    return kgcnn_backend.scatter_reduce_softmax(indices, values, shape, normalize=normalize)
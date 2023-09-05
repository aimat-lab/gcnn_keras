from kgcnn import backend
from keras_core.backend import KerasTensor
from keras_core.backend import any_symbolic_tensors
from keras_core.ops.operation import Operation


class ScatterMax(Operation):
    def call(self, indices, values, shape):
        return backend.scatter_max(indices, values, shape)

    def compute_output_spec(self, indices, values, shape):
        return KerasTensor(shape, dtype=values.dtype)


def scatter_max(indices, values, shape):
    if any_symbolic_tensors((indices, values, shape)):
        return ScatterMax().symbolic_call(indices, values, shape)
    return backend.scatter_max(indices, values, shape)


class ScatterMin(Operation):
    def call(self, indices, values, shape):
        return backend.scatter_min(indices, values, shape)

    def compute_output_spec(self, indices, values, shape):
        return KerasTensor(shape, dtype=values.dtype)


def scatter_min(indices, values, shape):
    if any_symbolic_tensors((indices, values, shape)):
        return ScatterMin().symbolic_call(indices, values, shape)
    return backend.scatter_min(indices, values, shape)


class ScatterMean(Operation):
    def call(self, indices, values, shape):
        return backend.scatter_mean(indices, values, shape)

    def compute_output_spec(self, indices, values, shape):
        return KerasTensor(shape, dtype=values.dtype)


def scatter_mean(indices, values, shape):
    if any_symbolic_tensors((indices, values, shape)):
        return ScatterMean().symbolic_call(indices, values, shape)
    return backend.scatter_mean(indices, values, shape)
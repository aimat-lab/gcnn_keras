import numpy as np
from keras_core import KerasTensor
from keras_core import Operation
from kgcnn.backend import any_symbolic_tensors
import kgcnn.backend as kgcnn_backend


class _RepeatStaticLength(Operation):

    def __init__(self, total_repeat_length: int, axis=None):
        super().__init__()
        self.axis = axis
        self.total_repeat_length = total_repeat_length

    def call(self, x, repeats):
        return kgcnn_backend.repeat_static_length(
            x, repeats, total_repeat_length=self.total_repeat_length, axis=self.axis)

    def compute_output_spec(self, x, repeats):
        output_shape = list(x.shape)
        if self.axis is None:
            return KerasTensor([self.total_repeat_length], dtype=x.dtype)
        output_shape[self.axis] = self.total_repeat_length
        return KerasTensor(output_shape, dtype=x.dtype)


def repeat_static_length(x, repeats, total_repeat_length: int, axis=None):
    """Repeat each element of a tensor after themselves.

    Args:
        x: Input tensor.
        repeats: The number of repetitions for each element.
        total_repeat_length: length of all repeats.
        axis: The axis along which to repeat values. By default, use
            the flattened input array, and return a flat output array.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x, repeats)):
        return _RepeatStaticLength(axis=axis, total_repeat_length=total_repeat_length).symbolic_call(x, repeats)
    return kgcnn_backend.repeat_static_length(x, repeats, axis=axis, total_repeat_length=total_repeat_length)


class _DecomposeRaggedTensor(Operation):

    def __init__(self, batch_dtype: str = "int64"):
        super().__init__()
        self.batch_dtype = batch_dtype

    def call(self, x):
        kgcnn_backend.decompose_ragged_tensor(x)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        output_shape = tuple(x_shape[1:])
        length_shape = (None, ) if x_shape[0] is None else (x_shape[0], )
        id_shape = output_shape[:1]
        return (KerasTensor(output_shape, dtype=x.dtype), KerasTensor(id_shape, dtype="int64"),
                KerasTensor(id_shape, dtype="int64"), KerasTensor(length_shape, dtype="int64"))


def decompose_ragged_tensor(x, batch_dtype="int64"):
    """Decompose ragged tensor.

    Args:
        x: Input tensor (ragged).
        batch_dtype (str): Data type for batch information. Default is 'int64'.

    Returns:
        Output tensors.
    """
    if any_symbolic_tensors((x,)):
        return _DecomposeRaggedTensor(batch_dtype=batch_dtype).symbolic_call(x)
    return kgcnn_backend.decompose_ragged_tensor(x)
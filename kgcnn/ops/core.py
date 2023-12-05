import numpy as np
from keras import KerasTensor
from keras import Operation
from keras.backend import result_type
from kgcnn.backend import any_symbolic_tensors
import kgcnn.backend as kgcnn_backend
from kgcnn.ops.axis import broadcast_shapes


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


def _reduce_shape(shape, axis=None, keepdims=False):
    shape = list(shape)
    if axis is None:
        if keepdims:
            return tuple([1 for _ in shape])
        else:
            return tuple([])

    if keepdims:
        for ax in axis:
            shape[ax] = 1
        return tuple(shape)
    else:
        for ax in sorted(axis, reverse=True):
            del shape[ax]
        return tuple(shape)


class _Norm(Operation):
    def __init__(self, ord='fro', axis=None, keepdims=False):
        super().__init__()
        self.ord = ord
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.keepdims = keepdims

    def call(self, x):
        return kgcnn_backend.norm(x, ord=self.ord, axis=self.axis, keepdims=self.keepdims)

    def compute_output_spec(self, x):
        return KerasTensor(
            _reduce_shape(x.shape, axis=self.axis, keepdims=self.keepdims),
            dtype=x.dtype,
        )


def norm(x, ord='fro', axis=None, keepdims=False):
    """Compute linalg norm.

    Args:
        x: Input tensor
        ord: Order of the norm.
        axis: dimensions over which to compute the vector or matrix norm.
        keepdims: If set to True, the reduced dimensions are retained in the result.

    Returns:
        output tensor.
    """
    if any_symbolic_tensors((x,)):
        return _Norm(ord=ord, axis=axis, keepdims=keepdims).symbolic_call(x)
    return kgcnn_backend.norm(x, ord=ord, axis=axis, keepdims=keepdims)


class _Cross(Operation):
    def __init__(self):
        super().__init__()
        self.axis = -1

    def call(self, x1, x2):
        return kgcnn_backend.cross(x1, x2)

    def compute_output_spec(self, x1, x2):
        x1_shape = list(x1.shape)
        x2_shape = list(x2.shape)

        x1_value_size = x1_shape[self.axis]
        x2_value_size = x2_shape[self.axis]
        del x1_shape[self.axis]
        del x2_shape[self.axis]
        output_shape = broadcast_shapes(x1_shape, x2_shape)

        if x1_value_size is not None and x1_value_size not in (2, 3):
            raise ValueError(
                "`x1`'s dim on `axis={axisa}` must be either 2 or 3, but "
                f"received: {x1_value_size}"
            )
        if x2_value_size is not None and x2_value_size not in (2, 3):
            raise ValueError(
                "`x2`'s dim on `axis={axisb}` must be either 2 or 3, but "
                f"received: {x2_value_size}"
            )

        if x1_value_size == 3 or x2_value_size == 3:
            value_size = [3]
        else:
            value_size = []

        output_shape = (
            output_shape[: self.axis] + value_size + output_shape[self.axis :]
        )
        dtype = result_type(x1.dtype, x2.dtype)
        return KerasTensor(output_shape, dtype=dtype)


def cross(x1, x2):
    """Returns the cross product of two (arrays of) vectors.

    Args:
        x1: Components of the first vector(s).
        x2: Components of the second vector(s).

    Returns:
        Vector cross product(s).
    """
    if any_symbolic_tensors((x1, x2)):
        return _Cross().symbolic_call(x1, x2)
    return kgcnn_backend.cross(x1, x2)

from keras_core.ops import any_symbolic_tensors
from keras_core.ops.numpy import Repeat
import kgcnn.backend as kgcnn_backend


def repeat_static_length(x, repeats, axis=None, total_repeat_length: int = None):
    """Repeat each element of a tensor after themselves.

    Args:
        x: Input tensor.
        repeats: The number of repetitions for each element.
        axis: The axis along which to repeat values. By default, use
            the flattened input array, and return a flat output array.
        total_repeat_length: length of all repeats.

    Returns:
        Output tensor.
    """
    if any_symbolic_tensors((x,)):
        return Repeat(repeats, axis=axis).symbolic_call(x)
    return kgcnn_backend.repeat_static_length(x, repeats, axis=axis, total_repeat_length=total_repeat_length)

import numpy as np
import tensorflow as tf

from kgcnn.utils.data import ragged_tensor_from_nested_numpy

class NumpyTensorList:

    def __init__(self, *args):
        self._tensor_list = args
        self._data_length = None
        # check that all

    def __getitem__(self, item):
        if isinstance(item, int):
            return NumpyTensorList(*[[x[item]] for x in self._tensor_list])
        elif isinstance(item, slice):
            return NumpyTensorList(*[x[item] for x in self._tensor_list])
        elif hasattr(item, "__getitem__"):
            return NumpyTensorList(*[[x[y] for y in item] for x in self._tensor_list])
        else:
            raise ValueError("Can not iterate over input of getitem() for `NumpyTensorList`: ", item)

    def tensor(self, ragged=None):
        if ragged is None:
            ragged = [False]*len(self._tensor_list)
        assert len(ragged) == len(self._tensor_list)
        out_list = []
        for i, x in enumerate(self._tensor_list):
            if not ragged[i]:
                out_list.append(tf.constant(x))
            else:
                out_list.append(ragged_tensor_from_nested_numpy(x))
        return out_list

# test = NumpyTensorList([np.array([1, 2])], [np.array([1, 2])])

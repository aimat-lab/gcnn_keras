import numpy as np
import tensorflow as tf

from kgcnn.utils.data import ragged_tensor_from_nested_numpy


class NumpyTensorList:
    r"""An utility class to keep lists of numpy arrays. Multiple lists are handled in parallel and therefore should
    have the same size. The behavior should be similar to :obj:`train_test_split` from scikit-learn.
    The length of each list should match the length of the dataset and
    each list-item be a numpy-tensor for a data-point. The lists could be X, y or different graph attributes.
    This class is intentionally kept simple but can be built on in the future.

    """

    def __init__(self, *args):
        """Make a new `NumpyTensorList` object with list of matching data like features and labels of a dataset.

        .. code-block:: python

            import numpy as np
            data = NumpyTensorList([np.array([0,0,0]), np.array([0])], [np.array([1]), np.array([1,1])])
            print(data[0].tensor())

        Args:
            args: Multiple lists of np.ndarrays.
        """
        self._tensor_list = [x for x in args]
        if len(self._tensor_list) <= 0:
            print("WARNING:kgcnn: Received empty list input for `NumpyTensorList`. \
                Expected one or more list of numpy arrays.")

        # check that all have same length
        self._length_test()

    def _length_test(self):
        length_test = np.array([len(x) for x in self._tensor_list], dtype="int")
        if len(length_test) > 0:
            if not np.all(length_test == length_test[0]):
                print("WARNING:kgcnn: Length of list input to `NumpyTensorList` are different.")

    def __getitem__(self, item):
        r"""Indexing or getitem method to apply to each list and return a new :obj:`NumpyTensorList`.

        Args:
            item (list, int, slice, np.ndarray): Index or list of indices to collect from each list in self.

        Returns:
            NumpyTensorList: New :obj:`NumpyTensorList` with only `items` in each list.
        """
        if isinstance(item, int):
            return NumpyTensorList(*[[x[item]] for x in self._tensor_list])
        elif isinstance(item, slice):
            return NumpyTensorList(*[x[item] for x in self._tensor_list])
        elif hasattr(item, "__getitem__"):
            return NumpyTensorList(*[[x[y] for y in item] for x in self._tensor_list])
        else:
            raise ValueError("ERROR:kgcnn: Can not iterate over input of getitem() for `NumpyTensorList`: ", item)

    def tensor(self, ragged=None):
        """Cast list of numpy arrays to tf.Tensor or tf.RaggedTensor objects.

        Note: A copy of the data may be generated for ragged tensors!

        Args:
            ragged (list): Information whether a output tensor for corresponding list is ragged. Default is None.

        Returns:
            list: A list of tf.Tensor or tf.RaggedTensor objects.
        """
        # TODO: We can check if tf.RaggedTensor is actually necessary, without relying on ragged-argument.
        # TODO: Also need to improve casting, e.g. memory usage
        # TODO: Also ragged_tensor_from_nested_numpy will fail for arbitrary ragged shape
        if ragged is None:
            ragged = [False] * len(self._tensor_list)
        assert len(ragged) == len(self._tensor_list)
        out_list = []
        for i, x in enumerate(self._tensor_list):
            if not ragged[i]:
                out_list.append(tf.constant(x))
            else:
                out_list.append(ragged_tensor_from_nested_numpy(x))
        return out_list

    def pop(self, index: int):
        r"""Remove a single item at index from each list within `NumpyTensorList`.

        Args:
            index (int): Index of item to remove.

        Returns:
            list: A list of removed items. Number of lists stored by `NumpyTensorList`.
        """
        removed_items = []
        for i, x in enumerate(self._tensor_list):
            if isinstance(x, list):
                temp = x.pop(index)
                removed_items.append(temp)
            else:
                removed_items.append(x[index])
                self._tensor_list[i] = [x[j] for j in range(len(x)) if j != index]
        return removed_items

    def __len__(self):
        """Get length of each list."""
        self._length_test()
        return len(self._tensor_list[0])


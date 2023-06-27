import os.path
import numpy as np
import tensorflow as tf
import h5py
from typing import List, Union


def _check_for_inner_shape(array_list: List[np.ndarray]) -> Union[None, tuple, list]:
    """Simple function to verify inner shape for list of numpy arrays."""
    # Cannot find inner shape for empty list.
    if len(array_list) == 0:
        return None
    # For fast check all items must be numpy arrays to get the inner shape easily.
    if not all(isinstance(x, np.ndarray) for x in array_list):
        return None
    shapes = [x.shape for x in array_list]
    # Must have all same rank.
    if not all(len(x) == len(shapes[0]) for x in shapes):
        return None
    # All Empty. No inner shape.
    if len(shapes[0]) == 0:
        return None
    # Empty inner shape.
    if len(shapes[0]) <= 1:
        return tuple([])
    # If all same inner shape.
    if all(x[1:] == shapes[0][1:] for x in shapes):
        return shapes[0][1:]


class RaggedTensorNumpyFile:
    """Class representing a NumPy '.npz' file to store a ragged tensor on disk.

    For the moment only ragged tensors of ragged rank of one are supported. However, arbitrary ragged tensors can be
    supported in principle.
    """

    _device = '/cpu:0'

    def __init__(self, file_path: str, compressed: bool = False):
        """Make class for a NPZ file.

        Args:
            file_path (str): Path to file on disk.
            compressed (bool): Whether to use compression.
        """
        self.file_path = file_path
        self.compressed = compressed

    def write(self, ragged_array: Union[tf.RaggedTensor, List[np.ndarray], list]):
        """Write ragged array to file.

        .. code-block:: python

            from kgcnn.io.file import RaggedTensorNumpyFile
            import numpy as np
            data = [np.array([[0, 1],[0, 2]]), np.array([[1, 1]]), np.array([[0, 1],[2, 2], [0, 3]])]
            f = RaggedTensorNumpyFile("test.npz")
            f.write(data)
            print(f.read())

        Args:
            ragged_array (list, tf.RaggedTensor): List or list of numpy arrays.

        Returns:
            None.
        """
        # We use tensorflow functions to ensure an eager ragged tensor.
        if not isinstance(ragged_array, tf.RaggedTensor):
            with tf.device(self._device):
                ragged_array = tf.ragged.constant(ragged_array, inner_shape=_check_for_inner_shape(ragged_array))
        assert ragged_array.ragged_rank == 1, "Only support for ragged_rank=1 at the moment."
        values = np.array(ragged_array.values)
        row_splits = np.array(ragged_array.row_splits)
        # Since the shape array can not have nones, we convert nones to 0.
        # Not ideal, but could make an extra shape array to indicate ragged dimensions.
        shape = np.array([x if x is not None else 0 for x in ragged_array.shape], dtype="uint64")
        ragged_rank = np.array(ragged_array.ragged_rank)
        rank = np.array(len(shape))
        out = {"values": values,
               "row_splits": row_splits,
               "shape": shape,
               "ragged_rank": ragged_rank,
               "rank": rank}
        if self.compressed:
            np.savez_compressed(self.file_path, **out)
        else:
            np.savez(self.file_path, **out)

    def read(self, return_as_tensor: bool = False):
        """Read the file into memory.

        Args:
            return_as_tensor: Whether to return tf.RaggedTensor.

        Returns:
            tf.RaggedTensor: Ragged tensor form file.
        """
        # Here only ragged rank one loading is supported.
        data = np.load(self.file_path)
        values = data.get("values")
        row_splits = data.get("row_splits")
        if return_as_tensor:
            with tf.device(self._device):
                out = tf.RaggedTensor.from_row_splits(values, row_splits)
            return out
        return np.split(values, row_splits[1:-1])

    def __getitem__(self, item):
        """Get single item from the ragged tensor on file.

        Args:
            item (int): Index of the item to get.
        """
        assert isinstance(item, int), "Only single index is supported, no slicing."
        # NOTE: At the moment mmap is not supported for NPZ files.
        with np.load(self.file_path, mmap_mode="r") as data:
            row_splits = np.array(data.get("row_splits"))
            out_data = np.array(data["values"][row_splits[item]:row_splits[item + 1]])
        return out_data

    def exists(self):
        """Check if file for path information of this class exists."""
        return os.path.exists(self.file_path)

    def __len__(self):
        """Length of the tensor on file."""
        data = np.load(self.file_path)
        row_splits = data.get("row_splits")
        num_row_splits = int(row_splits.shape[0])
        # length is num_row_splits - 1
        return num_row_splits-1


class RaggedTensorHDFile:
    """Class representing an HDF '.hdf5' file to store a ragged tensor on disk.

    For the moment only ragged tensors of ragged rank of one are supported. However, arbitrary ragged tensors can be
    supported in principle.
    """

    _device = '/cpu:0'

    def __init__(self, file_path: str, compressed: bool = None):
        """Make class for a HDF5 file.

        Args:
            file_path (str): Path to file on disk.
            compressed: Compression to use. Not used at the moment.
        """
        self.file_path = file_path
        self.compressed = compressed

    def write(self, ragged_array: List[np.ndarray]):
        """Write ragged array to file.

        .. code-block:: python

            from kgcnn.io.file import RaggedTensorHDFile
            import numpy as np
            data = [np.array([[0, 1],[0, 2]]), np.array([[1, 1]]), np.array([[0, 1],[2, 2], [0, 3]])]
            f = RaggedTensorHDFile("test.hdf5")
            f.write(data)
            print(f.read())

        Args:
            ragged_array (list, tf.RaggedTensor): List or list of numpy arrays.

        Returns:
            None.
        """
        # We use tensorflow functions to ensure an eager ragged tensor.
        if not isinstance(ragged_array, tf.RaggedTensor):
            with tf.device(self._device):
                ragged_array = tf.ragged.constant(ragged_array, inner_shape=_check_for_inner_shape(ragged_array))
        assert ragged_array.ragged_rank == 1, "Only support for ragged_rank=1 at the moment."
        values = np.array(ragged_array.values)
        row_splits = np.array(ragged_array.row_splits)
        # Since the shape array can not have nones, we convert nones to 0.
        # Not ideal, but could make an extra shape array to indicate ragged dimensions.
        shape = np.array([x if x is not None else 0 for x in ragged_array.shape], dtype="uint64")
        ragged_rank = np.array(ragged_array.ragged_rank)
        rank = np.array(len(shape))
        with h5py.File(self.file_path, "w") as file:
            file.create_dataset("values", data=values,
                                maxshape=[x if i > 0 else None for i, x in enumerate(values.shape)])
            file.create_dataset("row_splits", data=row_splits, maxshape=(None, ))
            file.create_dataset("shape", data=shape)
            file.create_dataset("rank", data=rank)
            file.create_dataset("ragged_rank", data=ragged_rank)

    def read(self, return_as_tensor: bool = False):
        """Read the file into memory.

        Args:
            return_as_tensor: Whether to return tf.RaggedTensor.

        Returns:
            tf.RaggedTensor: Ragged tensor form file.
        """
        with h5py.File(self.file_path, "r") as file:
            values = file["values"]
            row_splits = file["row_splits"]
            if return_as_tensor:
                with tf.device(self._device):
                    out = tf.RaggedTensor.from_row_splits(np.array(values), np.array(row_splits))
            else:
                out = np.split(values, row_splits[1:-1])
        return out

    def __getitem__(self, item: int):
        """Get single item from the ragged tensor on file.

        Args:
            item (int): Index of the item to get.
        """
        assert isinstance(item, int), "Only single index is supported, no slicing."
        with h5py.File(self.file_path, "r") as file:
            row_splits = file["row_splits"]
            out_data = np.array(file["values"][row_splits[item]:row_splits[item+1]])
        return out_data

    def append(self, item):
        """Append single item to ragged tensor.

        Args:
            item (np.ndarray, tf.Tensor): Item to append.

        Returns:
            None.
        """
        with h5py.File(self.file_path, "r+") as file:
            file["values"].resize(
                file["values"].shape[0] + len(item), axis=0
            )
            split_last = file["row_splits"][-1]
            file["row_splits"].resize(
                file["row_splits"].shape[0] + 1, axis=0
            )
            len_last = len(item)
            file["row_splits"][-1] = split_last + len_last
            file["values"][split_last:split_last+len_last] = item

    def append_multiple(self, items: list):
        """Append multiple items to ragged tensor.

        Args:
            items (list): List of items to append. Must match in shape.

        Returns:
            None.
        """
        new_values = np.concatenate(items, axis=0)
        new_len = len(items)
        new_splits = np.cumsum([len(x) for x in items])
        with h5py.File(self.file_path, "r+") as file:
            file["values"].resize(
                file["values"].shape[0] + new_values.shape[0], axis=0
            )
            split_last = file["row_splits"][-1]
            file["row_splits"].resize(
                file["row_splits"].shape[0] + new_len, axis=0
            )
            file["row_splits"][-new_len:] = split_last + new_splits
            file["values"][split_last:+split_last+new_splits[-1]] = new_values

    def __len__(self):
        """Length of the tensor on file."""
        with h5py.File(self.file_path, "r") as file:
            num_row_splits = int(file["row_splits"].shape[0])
        # length is num_row_splits - 1
        return num_row_splits-1

    def exists(self):
        """Check if file for path information of this class exists."""
        return os.path.exists(self.file_path)

import os.path

import numpy as np
import h5py
from typing import List, Union


class RaggedTensorNumpyFile:

    def __init__(self, file_path: str, compressed: bool = False):
        self.file_path = file_path
        self.compressed = compressed

    def write(self, ragged_array: List[np.ndarray]):
        inner_shape = ragged_array[0].shape
        values = np.concatenate([x for x in ragged_array], axis=0)
        row_splits = np.cumsum(np.array([len(x) for x in ragged_array], dtype="int64"), dtype="int64")
        row_splits = np.pad(row_splits, [1, 0])
        out = {"values": values, "row_splits": row_splits, "shape": np.array([])}
        if self.compressed:
            np.savez_compressed(self.file_path, **out)
        else:
            np.savez(self.file_path, **out)

    def read(self):
        data = np.load(self.file_path)
        values = data.get("values")
        row_splits = data.get("row_splits")
        return np.split(values, row_splits[1:-1])

    def __getitem__(self, item):
        raise NotImplementedError("Not implemented for file reference load.")


class RaggedTensorHDFile:

    def __init__(self, file_path: str, compressed: bool = None):
        self.file_path = file_path
        self.compressed = compressed

    def write(self, ragged_array: List[np.ndarray]):
        """Write ragged array to file.

        .. code-block:: python

            from kgcnn.io.file import RaggedArrayHDFile
            import numpy as np
            data = [np.array([[0, 1],[0, 2]]), np.array([[1, 1]]), np.array([[0, 1],[2, 2], [0, 3]])]
            f = RaggedArrayHDFile("test.hdf5")
            f.write(data)
            print(f.read())

        Args:
            ragged_array (list, tf.RaggedTensor): List or list of numpy arrays.

        Returns:
            None.
        """
        inner_shape = ragged_array[0].shape
        values = np.concatenate([x for x in ragged_array], axis=0)
        row_splits = np.cumsum(np.array([len(x) for x in ragged_array], dtype="int64"), dtype="int64")
        row_splits = np.pad(row_splits, [1, 0])
        with h5py.File(self.file_path, "w") as file:
            file.create_dataset("values", data=values, maxshape=[None] + list(inner_shape)[1:])
            file.create_dataset("row_splits", data=row_splits, maxshape=(None, ))
            file.create_dataset("shape", data=np.array([]))

    def read(self):
        with h5py.File(self.file_path, "r") as file:
            data = np.split(file["values"][()], file["row_splits"][1:-1])
        return data

    def __getitem__(self, item: int):
        with h5py.File(self.file_path, "r") as file:
            row_splits = file["row_splits"]
            out_data = file["values"][row_splits[item]:row_splits[item+1]]
        return out_data

    def append(self, item):
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
        with h5py.File(self.file_path, "r") as file:
            num_row_splits = file["row_splits"].shape[0]
        # length is num_row_splits - 1
        return num_row_splits-1

    def exists(self):
        return os.path.exists(self.file_path)

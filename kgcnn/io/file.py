import numpy as np
import h5py


class RaggedArrayNumpyFile:

    def __init__(self, file_path: str, compressed: bool = False):
        self.file_path = file_path
        self.compressed = compressed

    def write(self, ragged_array: list):
        # Only support ragged one for the moment.
        leading_shape = ragged_array[0].shape
        assert all([leading_shape[1:] == x.shape[1:] for x in ragged_array]), "Only support ragged rank == 1."
        values = np.concatenate([x for x in ragged_array], axis=0)
        row_splits = np.cumsum(np.array([len(x) for x in ragged_array], dtype="int64"), dtype="int64")
        out = {"values": values, "row_splits": row_splits}
        if self.compressed:
            np.savez_compressed(self.file_path, **out)
        else:
            np.savez(self.file_path, **out)

    def read(self):
        data = np.load(self.file_path)
        values = data.get("values")
        row_splits = data.get("row_splits")
        return np.split(values, row_splits[:-1])


class RaggedArrayHDFile:

    def __init__(self, file_path: str, compressed: bool = False):
        self.file_path = file_path
        self.compressed = compressed

    def write(self, ragged_array: list):
        values = np.concatenate([x for x in ragged_array], axis=0)
        row_splits = np.cumsum(np.array([len(x) for x in ragged_array], dtype="int64"), dtype="int64")
        with h5py.File("mytestfile.hdf5", "w") as f:
            f.create_dataset("values", data=values)
            f.create_dataset("row_splits", data=row_splits)

    def read(self):
        h5py.File(self.file_path)
        return None

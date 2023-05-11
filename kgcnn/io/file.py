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
        """Write ragged array to file.

        .. code-block:: python

            from kgcnn.io.file import RaggedArrayHDFile
            import numpy as np
            data = [np.array([[0, 1],[0, 2]]), np.array([[1, 1]]), np.array([[0, 1],[2, 2], [0, 3]])]
            f = RaggedArrayHDFile("test.hdf5")
            f.write(data)
            print(f.read())
            print(f[1])

        Args:
            ragged_array (list): List of numpy arrays.

        Returns:
            None.
        """
        values = np.concatenate([x for x in ragged_array], axis=0)
        row_splits = np.cumsum(np.array([len(x) for x in ragged_array], dtype="int64"), dtype="int64")
        with h5py.File(self.file_path, "w") as f:
            f.create_dataset("values", data=values)
            f.create_dataset("row_splits", data=row_splits)

    def read(self):
        file = h5py.File(self.file_path)
        data = np.split(file["values"][()], file["row_splits"][:1])
        file.close()
        return data

    def __getitem__(self, item):
        file = h5py.File(self.file_path)
        row_splits = file["row_splits"]
        row_splits = np.pad(row_splits, [1, 0])
        out_data = file["values"][row_splits[item]:row_splits[item+1]]
        file.close()
        return out_data

import numpy as np
import h5py


class RaggedArrayListNumpy:

    def __init__(self, file_path: str, compressed: bool = False):
        self.file_path = file_path
        self.compressed = compressed

    def write(self, array_list: list):
        out = {}

        if self.compressed:
            np.savez_compressed(self.file_path, **out)
        else:
            np.savez(self.file_path, **out)
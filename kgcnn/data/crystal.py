import os
import numpy as np

from kgcnn.data.base import MemoryGeometricGraphDataset
from kgcnn.utils.data import save_json_file
from kgcnn.mol.periodic import parse_cif_file_to_structures, convert_structures_as_dict


class CrystalDataset(MemoryGeometricGraphDataset):
    def __init__(self,
                 data_directory: str = None,
                 dataset_name: str = None,
                 file_name: str = None,
                 file_directory: str = None,
                 length: int = None,
                 verbose: int = 1, **kwargs):
        r"""Initialize a base class of :obj:`CrystalDataset`.

        Args:
            data_directory (str): Full path to directory of the dataset. Default is None.
            file_name (str): Generic filename for dataset to read into memory like a 'cif' file. Default is None.
            file_directory (str): Name or relative path from :obj:`data_directory` to a directory containing sorted
                'cif' files. Default is None.
            dataset_name (str): Name of the dataset. Important for naming and saving files. Default is None.
            length (int): Length of the dataset, if known beforehand. Default is None.
            verbose (int): Print progress or info for processing, where 0 is silent. Default is 1.
        """
        super(CrystalDataset, self).__init__(data_directory=data_directory, dataset_name=dataset_name,
                                             file_name=file_name, verbose=verbose, length=length,
                                             file_directory=file_directory, **kwargs)

    def _get_pymatgen_file_name(self):
        """Try to determine a file name for the mol information to store."""
        return "".join(self.file_name.split(".")[:-1]) + ".pymatgen.json"

    def prepare_data(self):
        file_path = os.path.join(self.data_directory, self.file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError("ERROR:kgcnn: Can not find file for `CrystalDataset`.")

        structs = parse_cif_file_to_structures(file_path)
        dicts = convert_structures_as_dict(structs)

        out_path = os.path.join(self.data_directory, self._get_pymatgen_file_name())
        save_json_file(dicts, out_path)

    def read_in_memory(self):
        pass



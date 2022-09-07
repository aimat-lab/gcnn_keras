import os
import numpy as np
from typing import Union
from kgcnn.scaler.mol import QMGraphLabelScaler
from sklearn.preprocessing import StandardScaler
from kgcnn.data.base import MemoryGraphDataset
from kgcnn.mol.io import parse_list_to_xyz_str, read_xyz_file, \
    write_mol_block_list_to_sdf, read_mol_list_from_sdf_file, write_list_to_xyz_file
from kgcnn.mol.module_babel import convert_xyz_to_mol_openbabel, MolecularGraphOpenBabel
from kgcnn.data.utils import pandas_data_frame_columns_to_numpy
from kgcnn.mol.methods import global_proton_dict, inverse_global_proton_dict


class QMDataset(MemoryGraphDataset):
    r"""This is a base class for 'quantum mechanical' datasets. It generates graph properties from a xyz-file, which
    stores atomic coordinates.

    Additionally, it should be possible to generate approximate chemical bonding information via `openbabel`, if this
    additional package is installed.
    The class inherits :obj:`MemoryGraphDataset`.

    At the moment, there is no connection to :obj:`MoleculeNetDataset` since usually for geometric data, the usage is
    related to learning quantum properties like energy, orbitals or forces and no "chemical" feature information is
    required.
    """

    _global_proton_dict = global_proton_dict
    _inverse_global_proton_dict = inverse_global_proton_dict

    def __init__(self, data_directory: str = None, dataset_name: str = None, file_name: str = None,
                 verbose: int = 10, file_directory: str = None):
        r"""Default initialization. File information on the location of the dataset on disk should be provided here.

        Args:
            data_directory (str): Full path to directory of the dataset. Default is None.
            file_name (str): Filename for reading into memory. This must be the base-name of a '.xyz' file.
                Or additionally the name of a '.csv' formatted file that has a list of file names.
                Files are expected to be in :obj:`file_directory`. Default is None.
            file_directory (str): Name or relative path from :obj:`data_directory` to a directory containing sorted
                '.xyz' files. Only used if :obj:`file_name` is None. Default is None.
            dataset_name (str): Name of the dataset. Important for naming and saving files. Default is None.
            verbose (int): Logging level. Default is 10.
        """
        MemoryGraphDataset.__init__(self, data_directory=data_directory, dataset_name=dataset_name,
                                    file_name=file_name, verbose=verbose,
                                    file_directory=file_directory)
        self.label_units = None
        self.label_names = None

    @classmethod
    def _make_mol_list(cls, atoms_coordinates_xyz: list):
        """Make mol-blocks from list of multiple molecules.

        Args:
            atoms_coordinates_xyz (list): Nested list of xyz information for each molecule such as
                `[[['C', 'H', ... ], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], ... ]], ... ]`.

        Returns:
            list: A list of mol-blocks as string.
        """
        mol_list = []
        for x in atoms_coordinates_xyz:
            xyz_str = parse_list_to_xyz_str(x)
            mol_str = convert_xyz_to_mol_openbabel(xyz_str)
            mol_list.append(mol_str)
        return mol_list

    def prepare_data(self, overwrite: bool = False, xyz_column_name: str = None, make_sdf: bool = True):
        r"""Pre-computation of molecular structure information in a sdf-file from a xyz-file or a folder of xyz-files.
        If there is no single xyz-file, it will be created with the information of a csv-file with the same name.

        Args:
            overwrite (bool): Overwrite existing database SDF file. Default is False.
            xyz_column_name (str): Name of the column in csv file with list of xyz-files located in file_directory
            make_sdf (bool): Whether to try to make a sdf file from xyz information via OpenBabel.

        Returns:
            self
        """
        xyz_list = None

        if os.path.exists(self.file_path_mol) and not overwrite:
            self.info("Found SDF-file %s of pre-computed structures." % self.file_path_mol)
            return self

        # Collect single xyz files in directory
        if not os.path.exists(self.file_path_xyz):
            self.read_in_table_file()

            if self.data_frame is None:
                raise FileNotFoundError("Can not find csv table with file names.")

            if xyz_column_name is None:
                raise ValueError("Please specify column for csv file which contains file names.")

            if xyz_column_name not in self.data_frame.columns:
                raise ValueError(
                    "Can not find file-names of column %s in %s" % (xyz_column_name, self.data_frame.columns))

            xyz_file_list = self.data_frame[xyz_column_name].values
            num_molecules = len(xyz_file_list)

            if not os.path.exists(os.path.join(self.data_directory, self.file_directory)):
                raise ValueError("No file directory of xyz files.")

            self.info("Read %s single xyz-files." % num_molecules)
            xyz_list = []
            for i, x in enumerate(xyz_file_list):
                # Only one file per path
                xyz_info = read_xyz_file(os.path.join(self.data_directory, self.file_directory, x))
                xyz_list.append(xyz_info[0])
                if i % 1000 == 0:
                    self.info("... Read structure {0} from {1}".format(i, num_molecules))
            # Make single file for later loading, which is faster.
            write_list_to_xyz_file(self.file_path_xyz, xyz_list)

        # Or the default is to read from single xyz-file.
        if xyz_list is None:
            self.info("Reading single xyz-file.")
            xyz_list = read_xyz_file(self.file_path_xyz)

        # Additionally, try to make SDF file
        if make_sdf:
            self.info("Converting xyz to mol information.")
            mb = self._make_mol_list(xyz_list)
            write_mol_block_list_to_sdf(mb, self.file_path_mol)
        return self

    @property
    def file_path_mol(self):
        """Try to determine a file name for the mol information to store."""
        return os.path.splitext(self.file_path)[0] + ".sdf"

    @property
    def file_path_xyz(self):
        """Try to determine a file name for the mol information to store."""
        return os.path.splitext(self.file_path)[0] + ".xyz"

    def read_in_memory(self, label_column_name: Union[str, list] = None):
        """Read xyz-file geometric information into memory. Optionally read also mol information. And try to find CSV
        file with graph labels if a column is specified by :obj:`label_column_name`.

        Returns:
            self
        """
        # Try to read xyz file here.
        xyz_list = read_xyz_file(self.file_path_xyz)
        symbol = [np.array(x[0]) for x in xyz_list]
        coord = [np.array(x[1], dtype="float")[:, :3] for x in xyz_list]
        nodes = [np.array([self._global_proton_dict[x] for x in y[0]], dtype="int") for y in xyz_list]

        self.assign_property("node_coordinates", coord)
        self.assign_property("node_symbol", symbol)
        self.assign_property("node_number", nodes)

        # Try also to read SDF file.
        self.read_in_memory_sdf()

        # Try also to read labels
        self.read_in_table_file()
        if self.data_frame is not None and label_column_name is not None:
            labels = pandas_data_frame_columns_to_numpy(self.data_frame, label_column_name)
            self.assign_property("graph_labels", [x for x in labels])
        return self

    def read_in_memory_sdf(self):
        """Read SDF-file with chemical structure information into memory.

        Returns:
            self
        """
        if not os.path.exists(self.file_path_mol):
            self.warning("Can not load SDF-file for dataset %s" % self.dataset_name)
            return self

        # Load sdf file here.
        mol_list = read_mol_list_from_sdf_file(self.file_path_mol)
        if mol_list is None:
            self.warning("Failed to load bond information from SDF file.")
            return self

        # Parse information
        self.info("Parsing mol information ...")
        bond_number = []
        edge_index = []
        edge_attr = []
        for x in mol_list:
            # Must not change number of atoms or coordinates here.
            mol = MolecularGraphOpenBabel().from_mol_block(x, keep_hs=True)
            if mol is None:
                bond_number.append(None)
                edge_index.append(None)
                continue

            temp_edge = mol.edge_number
            bond_number.append(np.array(temp_edge[1], dtype="int"))
            edge_index.append(np.array(temp_edge[0], dtype="int"))

        self.assign_property("edge_indices", edge_index)
        self.assign_property("edge_number", bond_number)
        return self

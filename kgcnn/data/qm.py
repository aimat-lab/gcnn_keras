import os
import numpy as np

from kgcnn.data.base import MemoryGraphDataset
from kgcnn.utils.adj import add_edges_reverse_indices
from kgcnn.mol.io import convert_list_to_xyz_str, read_xyz_file, \
    write_mol_block_list_to_sdf, parse_mol_str, dummy_load_sdf_file, write_list_to_xyz_file
from kgcnn.mol.openbabel import convert_xyz_to_mol_ob
from kgcnn.utils.data import pandas_data_frame_columns_to_numpy


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

    _global_proton_dict = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
                           'Na': 11,
                           'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
                           'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
                           'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38,
                           'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47,
                           'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56,
                           'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65,
                           'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
                           'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83,
                           'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
                           'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101,
                           'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
                           'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117,
                           'Og': 118, 'Uue': 119}
    _inverse_global_proton_dict = {value: key for key, value in _global_proton_dict.items()}

    def __init__(self, data_directory: str = None, dataset_name: str = None, file_name: str = None,
                 verbose: int = 1, length: int = None, file_directory: str = None):
        r"""Default initialization. File information on the location of the dataset on disk should be provided here.

        Args:
            data_directory (str): Full path to directory of the dataset. Default is None.
            file_name (str): Filename for reading into memory. This must be the base-name of a '.xyz' file.
                Or additionally the name of a '.csv' formatted file that has a list of file names.
                Files are expected to be in :obj:`file_directory`. Default is None.
            file_directory (str): Name or relative path from :obj:`data_directory` to a directory containing sorted
                '.xyz' files. Only used if :obj:`file_name` is None. Default is None.
            dataset_name (str): Name of the dataset. Important for naming and saving files. Default is None.
            length (int): Length of the dataset, if known beforehand. Default is None.
            verbose (int): Print progress or info for processing, where 0 is silent. Default is 1.
        """
        MemoryGraphDataset.__init__(self, data_directory=data_directory, dataset_name=dataset_name,
                                    file_name=file_name, verbose=verbose, length=length,
                                    file_directory=file_directory)

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
            xyz_str = convert_list_to_xyz_str(x)
            mol_str = convert_xyz_to_mol_ob(xyz_str)
            mol_list.append(mol_str)
        return mol_list

    def prepare_data(self, overwrite: bool = False, xyz_column_name: str = None):
        r"""Pre-computation of molecular structure information in a sdf-file from a xyz-file or a folder of xyz-files.
        If there is no single xyz-file, it will be created with the information of a csv-file with the same name.

        Args:
            overwrite (bool): Overwrite existing database SDF file. Default is False.
            xyz_column_name (str): Name of the column in csv file with list of xyz-files located in file_directory

        Returns:
            self
        """
        xyz_list = None

        # Names for single xyz and mol file.
        mol_file_path = os.path.join(self.data_directory, self._get_mol_filename())
        xyz_file_path = os.path.join(self.data_directory, self._get_xyz_filename())

        if os.path.exists(mol_file_path) and not overwrite:
            self.info("Found SDF-file %s of pre-computed structures." % mol_file_path)
            return self

        # Collect single xyz files in directory
        if not os.path.exists(xyz_file_path):
            self.read_in_table_file()

            if self.data_frame is None:
                raise FileNotFoundError("Can not find csv table with file names.")

            if xyz_column_name is None:
                raise ValueError("Please specify column for csv file which contains file names.")

            xyz_file_list = self.data_frame[xyz_column_name].values
            num_mols = len(xyz_file_list)

            if not os.path.exists(os.path.join(self.data_directory, self.file_directory)):
                raise ValueError("No file directory of xyz files.")

            self.info("Read %s single xyz-files ..." % num_mols)
            xyz_list = []
            for i, x in enumerate(xyz_file_list):
                # Only one file per path
                xyz_info = read_xyz_file(os.path.join(self.data_directory, self.file_directory, x))
                xyz_list.append(xyz_info[0])
                if i % 1000 == 0:
                    self.info(" ... read structure {0} from {1}".format(i, num_mols))
            # Make single file
            write_list_to_xyz_file(xyz_file_path, xyz_list)

        # Additionally try to make SDF file
        try:
            from openbabel import openbabel
        except ImportError:
            self.warning("Can not make mol-objects. Please install openbabel.")
            return self

        if xyz_list is None:
            self.info("Reading single xyz-file ...")
            filepath = os.path.join(self.data_directory, self.file_name)
            xyz_list = read_xyz_file(filepath)

        self.info("Converting xyz to mol information (silent)...")
        mb = self._make_mol_list(xyz_list)
        write_mol_block_list_to_sdf(mb, mol_file_path)
        return self

    def _get_mol_filename(self):
        """Try to determine a file name for the mol information to store."""
        return os.path.splitext(self.file_name)[0] + ".sdf"

    def _get_xyz_filename(self):
        """Try to determine a file name for the mol information to store."""
        return os.path.splitext(self.file_name)[0] + ".xyz"

    def read_in_memory(self, label_column_name: str = None):
        """Read xyz-file geometric information into memory. Optionally read also mol information. And try to find CSV
        file with graph labels if a column is specified by :obj:`label_column_name`.

        Returns:
            self
        """
        filepath = os.path.join(self.data_directory, self.file_name)

        # Try to read xyz file here.
        xyz_list = read_xyz_file(filepath)
        symbol = [np.array([x[0] for x in y]) for y in xyz_list]
        coords = [np.array([x[1:4] for x in y], dtype="float") for y in xyz_list]
        nodes = [np.array([self._global_proton_dict[x[0]] for x in y], dtype="int") for y in xyz_list]
        self.length = len(symbol)
        self.node_coordinates = coords
        self.node_symbol = symbol
        self.node_number = nodes

        # Try also to read SDF file.
        self.read_in_memory_sdf()

        # Try also to read labels
        self.read_in_table_file(file_path=filepath)

        # We can try to get labels here.
        if self.data_frame is not None and label_column_name is not None:
            self.graph_labels = pandas_data_frame_columns_to_numpy(self.data_frame, label_column_name)
        return self

    def read_in_memory_sdf(self):
        """Read SDF-file with chemical structure information into memory.

        Returns:
            self
        """
        mol_filename = self._get_mol_filename()
        mol_path = os.path.join(self.data_directory, mol_filename)

        if not os.path.exists(mol_path):
            self.warning("Can not load SDF-file for dataset %s" % self.dataset_name)
            return self

        # Load sdf file here.
        mol_list = dummy_load_sdf_file(mol_path)
        if mol_list is not None:
            self.info("Parsing mol information ...")
            bond_info = []
            for x in mol_list:
                bond_block = parse_mol_str(x)[5]
                bond_array = np.array([[int(z) for z in y] for y in bond_block], dtype="int")
                bond_info.append(bond_array)
            edge_index = []
            edge_attr = []
            for x in bond_info:
                if len(x) == 0:
                    edge_index.append(np.array([[]], dtype="int"))
                    edge_attr.append(np.array([[]], dtype="float"))
                else:
                    temp = add_edges_reverse_indices(np.array(x[:, :2]), np.array(x[:, 2:]))
                    edge_index.append(temp[0] - 1)
                    edge_attr.append(np.array(temp[1], dtype="float"))
            self.edge_indices = edge_index
            self.edge_attributes = edge_attr
        return self

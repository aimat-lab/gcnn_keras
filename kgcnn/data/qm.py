import os
import numpy as np
import logging

from kgcnn.mol.methods import ExtensiveMolecularScaler
from sklearn.preprocessing import StandardScaler
from kgcnn.data.base import MemoryGraphDataset
from kgcnn.mol.io import parse_list_to_xyz_str, read_xyz_file, \
    write_mol_block_list_to_sdf, parse_mol_str, read_mol_list_from_sdf_file, write_list_to_xyz_file
from kgcnn.mol.graphBabel import convert_xyz_to_mol_openbabel, MolecularGraphOpenBabel
from kgcnn.utils.data import pandas_data_frame_columns_to_numpy
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

        # Additionally try to make SDF file
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

    def read_in_memory(self, label_column_name: str = None):
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
            mol = MolecularGraphOpenBabel(add_hydrogen=False, make_conformer=False,
                                          optimize_conformer=False).from_mol_block(x)
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


class QMGraphLabelScaler:
    """A scaler that scales QM targets differently. For now, the main difference is that intensive and extensive
    properties are scaled differently. In principle, also dipole, polarizability or rotational constants
    could to be standardized differently.

    """

    def __init__(self, scaler: list):
        if not isinstance(scaler, list):
            raise TypeError("Scaler information for `QMGraphLabelScaler` must be list, got %s." % scaler)

        self.scaler_list = []
        for x in scaler:
            # If x is already a scaler, add it directly to the scaler list.
            if hasattr(x, "fit") and hasattr(x, "transform") and hasattr(x, "inverse_transform"):
                self.scaler_list.append(x)
                continue
            # Otherwise must be serialized version of a scaler.
            if not isinstance(x, dict):
                raise TypeError("Single scaler for `QMGraphLabelScaler` must be dict, got %s." % x)

            if "class_name" not in x:
                raise ValueError("Scaler class for single target must be defined, got %s" % x)

            if x["class_name"] == "StandardScaler":
                self.scaler_list.append(StandardScaler(**x["config"]))
            elif x["class_name"] == "ExtensiveMolecularScaler":
                self.scaler_list.append(ExtensiveMolecularScaler(**x["config"]))
            else:
                raise ValueError("Unsupported scaler %s" % x["name"])

        self.scale_ = None

    def _input_for_each_scaler_type(self, scaler, graph_labels, node_number):
        if isinstance(scaler, StandardScaler):
            return [graph_labels]
        elif isinstance(scaler, ExtensiveMolecularScaler):
            return node_number, graph_labels
        raise TypeError("Unsupported scaler %s" % scaler)

    def _scale_for_each_scaler_type(self, scaler):
        if isinstance(scaler, StandardScaler):
            return scaler.scale_
        elif isinstance(scaler, ExtensiveMolecularScaler):
            return scaler.scale_[0]
        raise TypeError("Unsupported scaler %s" % scaler)

    def fit_transform(self, graph_labels, node_number):
        r"""Fit and transform all target labels for QM9.

        Args:
            graph_labels (np.ndarray): Array of QM9 labels of shape `(N, 15)`.
            node_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.

        Returns:
            np.ndarray: Transformed labels of shape `(N, 15)`.
        """
        self.fit(graph_labels, node_number)
        return self.transform(graph_labels, node_number)

    def transform(self, graph_labels, node_number):
        r"""Transform all target labels for QM. Requires :obj:`fit()` called previously.

        Args:
            graph_labels (np.ndarray): Array of QM unscaled labels of shape `(N, #labels)`.
            node_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.

        Returns:
            np.ndarray: Transformed labels of shape `(N, #labels)`.
        """
        self._check_input(node_number, graph_labels)

        out_labels = []
        for i, x in enumerate(self.scaler_list):
            labels = graph_labels[:, i:i+1]
            out_labels.append(x.transform(*self._input_for_each_scaler_type(x, labels, node_number)))

        out_labels = np.concatenate(out_labels, axis=-1)
        return out_labels

    def fit(self, graph_labels, node_number):
        r"""Fit scaling of QM9 graph labels or targets.

        Args:
            graph_labels (np.ndarray): Array of QM labels of shape `(N, #labels)`.
            node_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.

        Returns:
            self
        """
        self._check_input(node_number, graph_labels)

        for i, x in enumerate(self.scaler_list):
            labels = graph_labels[:, i:i + 1]
            x.fit(*self._input_for_each_scaler_type(x, labels, node_number))

        self.scale_ = np.concatenate([self._scale_for_each_scaler_type(x) for x in self.scaler_list], axis=0)
        return self

    def inverse_transform(self, graph_labels, node_number):
        r"""Back-transform all target labels for QM9.

        Args:
            graph_labels (np.ndarray): Array of QM scaled labels of shape `(N, #labels)`.
            node_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.

        Returns:
            np.ndarray: Back-transformed labels of shape `(N, 15)`.
        """
        self._check_input(node_number, graph_labels)

        out_labels = []
        for i, x in enumerate(self.scaler_list):
            labels = graph_labels[:, i:i + 1]
            out_labels.append(x.inverse_transform(*self._input_for_each_scaler_type(x, labels, node_number)))

        out_labels = np.concatenate(out_labels, axis=-1)
        return out_labels

    def _check_input(self, node_number, graph_labels):
        assert len(node_number) == len(graph_labels), "`QMGraphLabelScaler` input length does not match."
        assert graph_labels.shape[-1] == len(self.scaler_list), "`QMGraphLabelScaler` got wrong number of labels."

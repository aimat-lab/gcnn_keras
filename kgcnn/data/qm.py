import os
import numpy as np
import pandas as pd
from typing import Union, Callable, List, Dict
from kgcnn.mol.base import MolGraphInterface
from kgcnn.scaler.mol import QMGraphLabelScaler
from sklearn.preprocessing import StandardScaler
from kgcnn.data.base import MemoryGraphDataset
from kgcnn.mol.io import parse_list_to_xyz_str, read_xyz_file, \
    write_mol_block_list_to_sdf, read_mol_list_from_sdf_file, write_list_to_xyz_file
from kgcnn.mol.methods import global_proton_dict, inverse_global_proton_dict
from kgcnn.data.moleculenet import MolGraphCallbacks

try:
    from openbabel import openbabel
    from kgcnn.mol.module_babel import convert_xyz_to_mol_openbabel, MolecularGraphOpenBabel
except ImportError:
    openbabel, convert_xyz_to_mol_openbabel, MolecularGraphOpenBabel = None, None, None

try:
    import rdkit
    from kgcnn.mol.module_rdkit import MolecularGraphRDKit
except ImportError:
    rdkit, MolecularGraphRDKit = None, None


class QMDataset(MemoryGraphDataset, MolGraphCallbacks):
    r"""This is a base class for QM (quantum mechanical) datasets.

    It generates graph properties from a xyz-file, which stores atomic coordinates.
    Additionally, loading multiple single xyz-files into one file is supported. The file names and labels are given
    by a CSV or table file. The table file must have one line of header with column names!

    .. code-block:: type

        ├── data_directory
            ├── file_directory
            │   ├── *.xyz
            │   ├── *.xyz
            │   └── ...
            ├── file_name.csv
            ├── file_name.xyz
            ├── file_name.sdf
            └── dataset_name.kgcnn.pickle

    It should be possible to generate approximate chemical bonding information via `openbabel`, if this
    additional package is installed. The class inherits from :obj:`MemoryGraphDataset` and :obj:`MolGraphCallbacks`.
    If `openbabel` is not installed minimal loading of labels and coordinates should be supported.

    For additional attributes, the :obj:`set_attributes` enables further features that require RDkit to be installed.

    """

    _global_proton_dict = global_proton_dict
    _inverse_global_proton_dict = inverse_global_proton_dict
    _default_loop_update_info = 5000

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

    @property
    def file_path_mol(self):
        """Try to determine a file name for the mol information to store."""
        return os.path.splitext(self.file_path)[0] + ".sdf"

    @property
    def file_path_xyz(self):
        """Try to determine a file name for the mol information to store."""
        return os.path.splitext(self.file_path)[0] + ".xyz"

    @classmethod
    def _convert_xyz_to_mol_list(cls, atoms_coordinates_xyz: list):
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

    def get_geom_from_xyz_file(self, file_path: str) -> list:
        """Get a list of xyz items from file.

        Args:
            file_path (str): File path of XYZ file. Default None uses :obj:`file_path_xyz`.

        Returns:
            list: List of xyz lists.
        """
        if file_path is None:
            file_path = self.file_path_xyz
        return read_xyz_file(file_path)

    def get_mol_blocks_from_sdf_file(self, file_path: str = None) -> list:
        """Get a list of mol-blocks from file.

        Args:
            file_path (str): File path of SDF file. Default None uses :obj:`file_path_mol`.

        Returns:
            list: List of mol-strings.
        """
        if file_path is None:
            file_path = self.file_path_mol
        if not os.path.exists(file_path):
            raise FileNotFoundError("Can not load SDF for dataset %s" % self.dataset_name)
        # Loading the molecules and the csv data
        mol_list = read_mol_list_from_sdf_file(file_path)
        if mol_list is None:
            self.warning("Failed to load bond information from SDF file.")
        return mol_list

    def prepare_data(self, overwrite: bool = False, file_column_name: str = None, make_sdf: bool = True):
        r"""Pre-computation of molecular structure information in a sdf-file from a xyz-file or a folder of xyz-files.

        If there is no single xyz-file, it will be created with the information of a csv-file with the same name.

        Args:
            overwrite (bool): Overwrite existing database SDF file. Default is False.
            file_column_name (str): Name of the column in csv file with list of xyz-files located in file_directory
            make_sdf (bool): Whether to try to make a sdf file from xyz information via OpenBabel.

        Returns:
            self
        """
        if os.path.exists(self.file_path_mol) and not overwrite:
            self.info("Found SDF file '%s' of pre-computed structures." % self.file_path_mol)
            return self

        # Try collect single xyz files in directory
        xyz_list = None
        if not os.path.exists(self.file_path_xyz):
            xyz_list = self.collect_files_in_file_directory(
                file_column_name=file_column_name, table_file_path=None,
                read_method_file=self.get_geom_from_xyz_file, update_counter=self._default_loop_update_info,
                append_file_content=True, read_method_return_list=True
            )
            write_list_to_xyz_file(self.file_path_xyz, xyz_list)

        # Additionally, try to make SDF file. Requires openbabel.
        if make_sdf:
            if xyz_list is None:
                self.info("Reading single xyz-file.")
                xyz_list = self.get_geom_from_xyz_file(self.file_path_xyz)
            self.info("Converting xyz to mol information.")
            write_mol_block_list_to_sdf(self._convert_xyz_to_mol_list(xyz_list), self.file_path_mol)
        return self

    def read_in_memory_xyz(self, file_path: str = None):
        """Read XYZ-file with geometric information into memory.

        Args:
            file_path (str): Filepath to xyz file.

        Returns:
            self
        """
        xyz_list = self.get_geom_from_xyz_file(file_path)
        symbol = [np.array(x[0]) for x in xyz_list]
        coord = [np.array(x[1], dtype="float")[:, :3] for x in xyz_list]
        nodes = [np.array([self._global_proton_dict[x] for x in y[0]], dtype="int") for y in xyz_list]
        self.assign_property("node_coordinates", coord)
        self.assign_property("node_symbol", symbol)
        self.assign_property("node_number", nodes)
        return self

    def read_in_memory_sdf(self, file_path: str = None, label_column_name: Union[str, list] = None):
        """Read SDF-file with chemical structure information into memory.

        Args:
            file_path (str): Filepath to SDF file.
            label_column_name (str, list): Name of labels for columns in CSV file.

        Returns:
            self
        """
        callbacks = {
            "node_symbol": lambda mg, ds: mg.node_symbol,
            "node_number": lambda mg, ds: mg.node_number,
            "edge_indices": lambda mg, ds: mg.edge_number[0],
            "edge_number": lambda mg, ds: np.array(mg.edge_number[1], dtype='int'),
        }
        if label_column_name:
            callbacks.update({'graph_labels': lambda mg, ds: ds[label_column_name]})

        self._map_molecule_callbacks(
            self.get_mol_blocks_from_sdf_file(file_path),
            self.read_in_table_file().data_frame,
            callbacks=callbacks,
            add_hydrogen=True,
            custom_transform=None,
            make_directed=False,
            mol_interface_class=MolecularGraphOpenBabel
        )
        return self

    def read_in_memory(self, label_column_name: Union[str, list] = None):
        """Read geometric information into memory.

        Graph labels require a column specified by :obj:`label_column_name`.

        Returns:
            self
        """
        if os.path.exists(self.file_path_mol) and openbabel is not None:
            self.read_in_memory_sdf(label_column_name=label_column_name)
        else:
            # 1. Read labels and xyz-file without openbabel.
            self.read_in_table_file()
            if self.data_frame is not None and label_column_name is not None:
                labels = self.data_frame[label_column_name]
                self.assign_property("graph_labels", [x for _, x in labels.iterrows()])
                self.read_in_memory_xyz()
        return self

    def set_attributes(self,
                       label_column_name: Union[str, list] = None,
                       nodes: list = None,
                       edges: list = None,
                       graph: list = None,
                       encoder_nodes: dict = None,
                       encoder_edges: dict = None,
                       encoder_graph: dict = None,
                       add_hydrogen: bool = False,
                       make_directed: bool = False,
                       has_conformers: bool = True,
                       additional_callbacks: Dict[str, Callable[[MolecularGraphRDKit, dict], None]] = None,
                       custom_transform: Callable[[MolecularGraphRDKit], MolecularGraphRDKit] = None):
        """Load list of molecules from cached SDF-file in into memory.

        File name must be given in :obj:`file_name` and path information in the constructor of this class.

        It further checks the csv-file for graph labels specified by :obj:`label_column_name`.
        Labels that do not have valid smiles or molecule in the SDF-file are also skipped, but added as `None` to
        keep the index and the molecule assignment.

        Set further molecular attributes or features by string identifier. Requires :obj:`MolecularGraphRDKit`.
        Default values are features that has been used by
        `Luo et al (2019) <https://doi.org/10.1021/acs.jmedchem.9b00959>`_.

        The argument :obj:`additional_callbacks` allows adding custom properties to each element of the dataset. It is
        a dictionary whose string keys are the names of the properties and the values are callable function objects
        which define how the property is derived from either the :obj:`MolecularGraphRDKit` or the corresponding
        row of the original CSV file. Those callback functions accept two parameters:

            * mg: The :obj:`MolecularGraphRDKit` instance of the molecule.
            * ds: A pandas data series that match data in the CSV file for the specific molecule.

        Args:
            label_column_name (str): Column name in the csv-file where to take graph labels from.
                For multi-targets you can supply a list of column names or positions. A slice can be provided
                for selecting columns as graph labels. Default is None.
            nodes (list): A list of node attributes as string. In place of names also functions can be added.
            edges (list): A list of edge attributes as string. In place of names also functions can be added.
            graph (list): A list of graph attributes as string. In place of names also functions can be added.
            encoder_nodes (dict): A dictionary of callable encoder where the key matches the attribute.
            encoder_edges (dict): A dictionary of callable encoder where the key matches the attribute.
            encoder_graph (dict): A dictionary of callable encoder where the key matches the attribute.
            add_hydrogen (bool): Whether to keep hydrogen after reading the mol-information. Default is False.
            has_conformers (bool): Whether to add node coordinates from conformer. Default is True.
            make_directed (bool): Whether to have directed or undirected bonds. Default is False.
            additional_callbacks (dict): A dictionary whose keys are string attribute names which the elements of the
                dataset are supposed to have and the elements are callback function objects which implement how those
                attributes are derived from the :obj:`MolecularGraphRDKit` of the molecule in question or the
                row of the CSV file.
            custom_transform (Callable): Custom transformation function to modify the generated
                :obj:`MolecularGraphRDKit` before callbacks are carried out. The function must take a single
                :obj:`MolecularGraphRDKit` instance as argument and return a (new) :obj:`MolecularGraphRDKit` instance.

        Returns:
            self
        """
        # May put this in a decorator with a copy or just leave as default arguments.
        # If e.g. nodes is not modified there is no problem with having mutable defaults.
        nodes = nodes if nodes is not None else self._default_node_attributes
        edges = edges if edges is not None else self._default_edge_attributes
        graph = graph if graph is not None else self._default_graph_attributes
        encoder_nodes = encoder_nodes if encoder_nodes is not None else self._default_node_encoders
        encoder_edges = encoder_edges if encoder_edges is not None else self._default_edge_encoders
        encoder_graph = encoder_graph if encoder_graph is not None else self._default_graph_encoders
        additional_callbacks = additional_callbacks if additional_callbacks is not None else {}

        # Deserializing encoders
        for encoder in [encoder_nodes, encoder_edges, encoder_graph]:
            for key, value in encoder.items():
                encoder[key] = self._deserialize_encoder(value)

        callbacks = {
            'node_symbol': lambda mg, ds: mg.node_symbol,
            'node_number': lambda mg, ds: mg.node_number,
            'edge_indices': lambda mg, ds: mg.edge_number[0],
            'edge_number': lambda mg, ds: np.array(mg.edge_number[1], dtype='int'),
            'graph_size': lambda mg, ds: len(mg.node_number)
        }
        if has_conformers:
            callbacks.update({'node_coordinates': lambda mg, ds: mg.node_coordinates})
        if label_column_name:
            callbacks.update({'graph_labels': lambda mg, ds: ds[label_column_name]})

        # Attributes callbacks.
        callbacks.update({
            'node_attributes': lambda mg, ds: np.array(mg.node_attributes(nodes, encoder_nodes), dtype='float32'),
            'edge_attributes': lambda mg, ds: np.array(mg.edge_attributes(edges, encoder_edges)[1], dtype='float32'),
            'graph_attributes': lambda mg, ds: np.array(mg.graph_attributes(graph, encoder_graph), dtype='float32')
        })

        # Additional callbacks. Could check for duplicate names here.
        callbacks.update(additional_callbacks)

        self._map_molecule_callbacks(
            self.get_mol_blocks_from_sdf_file(),
            self.read_in_table_file().data_frame,
            callbacks=callbacks,
            add_hydrogen=add_hydrogen,
            custom_transform=custom_transform,
            make_directed=make_directed,
            mol_interface_class=MolecularGraphRDKit
        )

        if self.logger.getEffectiveLevel() < 20:
            for encoder in [encoder_nodes, encoder_edges, encoder_graph]:
                for key, value in encoder.items():
                    if hasattr(value, "report"):
                        value.report(name=key)

        return self
import os
import numpy as np
import pandas as pd
from typing import Union, Callable, List, Dict
from kgcnn.molecule.base import MolGraphInterface
from kgcnn.data.transform.scaler.molecule import QMGraphLabelScaler
from sklearn.preprocessing import StandardScaler
from kgcnn.molecule.serial import deserialize_encoder
from kgcnn.data.base import MemoryGraphDataset
from kgcnn.molecule.io import parse_list_to_xyz_str, read_xyz_file, \
    write_mol_block_list_to_sdf, read_mol_list_from_sdf_file, write_list_to_xyz_file
from kgcnn.molecule.methods import global_proton_dict, inverse_global_proton_dict
from kgcnn.molecule.convert import MolConverter
from kgcnn.data.moleculenet import map_molecule_callbacks

try:
    from kgcnn.molecule.graph_babel import MolecularGraphOpenBabel
except ModuleNotFoundError:
    MolecularGraphOpenBabel = None

try:
    from kgcnn.molecule.graph_rdkit import MolecularGraphRDKit
except ModuleNotFoundError:
    MolecularGraphRDKit = None


class QMDataset(MemoryGraphDataset):
    r"""This is a base class for QM (quantum mechanical) datasets.

    It generates graph properties from a xyz-file, which stores atomic coordinates.
    Additionally, loading multiple single xyz-files into one file is supported. The file names and labels are given
    by a CSV or table file. The table file must have one line of header with column names!

    .. code-block:: console

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
    additional package is installed. The class inherits from :obj:`MemoryGraphDataset` .
    If `openbabel` is not installed minimal loading of labels and coordinates should be supported.

    For additional attributes, the :obj:`set_attributes` enables further features that require RDkit or Openbabel
    to be installed.
    Note that for QMDataset the mol-information, if it is generated, is not cleaned during reading by default.

    """

    _global_proton_dict = global_proton_dict
    _inverse_global_proton_dict = inverse_global_proton_dict
    _default_loop_update_info = 5000
    _mol_graph_interface = MolecularGraphRDKit  # other option is MolecularGraphOpenBabel

    def __init__(self, data_directory: str = None, dataset_name: str = None, file_name: str = None,
                 verbose: int = 10, file_directory: str = None, file_name_xyz: str = None, file_name_mol: str = None):
        r"""Default initialization. File information on the location of the dataset on disk should be provided here.

        Args:
            data_directory (str): Full path to directory of the dataset. Optional. Default is None.
            file_name (str): Filename for reading table '.csv' file into memory. Must be given!
                For example as '.csv' formatted file with QM labels such as energy, states, dipole etc.
                Moreover, the table file can contain a list of file names of individual '.xyz' files to collect.
                Files are expected to be in :obj:`file_directory`. Default is None.
            file_name_xyz (str): Filename of a single '.xyz' file. This file is generated when collecting single
                '.xyz' files in :obj:`file_directory` . If not specified, the name is generated based on
                :obj:`file_name` .
            file_name_mol (str): Filename of a single '.sdf' file. This file is generated from the single '.xyz'
                file. SDF generation does require proper geometries. If not specified, the name is generated based on
                :obj:`file_name` .
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
        self.file_name_xyz = file_name_xyz
        self.file_name_mol = file_name_mol

    @property
    def file_path_mol(self):
        """Try to determine a file name for the mol information to store."""
        self._verify_data_directory()
        if self.file_name_mol is None:
            return os.path.join(self.data_directory, os.path.splitext(self.file_name)[0] + ".sdf")
        else:
            return os.path.join(self.data_directory, self.file_name_mol)

    @property
    def file_path_xyz(self):
        """Try to determine a file name for the mol information to store."""
        self._verify_data_directory()
        if self.file_name_xyz is None:
            return os.path.join(self.data_directory, os.path.splitext(self.file_name)[0] + ".xyz")
        else:
            return os.path.join(self.data_directory, self.file_name_xyz)

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
            make_sdf (bool): Whether to try to make a sdf file from xyz information via `RDKit` and `OpenBabel`.

        Returns:
            self
        """
        if os.path.exists(self.file_path_mol) and not overwrite:
            self.info("Found SDF file '%s' of pre-computed structures." % self.file_path_mol)
            return self

        # Try collect single xyz files in directory
        if not os.path.exists(self.file_path_xyz):
            xyz_list = self.collect_files_in_file_directory(
                file_column_name=file_column_name, table_file_path=None,
                read_method_file=self.get_geom_from_xyz_file, update_counter=self._default_loop_update_info,
                append_file_content=True, read_method_return_list=True
            )
            write_list_to_xyz_file(self.file_path_xyz, xyz_list)

        # Additionally, try to make SDF file. Requires openbabel.
        if make_sdf:
            self.info("Converting xyz to mol information.")
            converter = MolConverter()
            converter.xyz_to_mol(self.file_path_xyz, self.file_path_mol)
        return self

    def read_in_memory_xyz(self, file_path: str = None,
                           atomic_coordinates: Union[str, None] = "node_coordinates",
                           atomic_symbol: Union[str, None] = "node_symbol",
                           atomic_number: Union[str, None] = "node_number"
                           ):
        """Read XYZ-file with geometric information into memory.

        Args:
            file_path (str): Filepath to xyz file.
            atomic_coordinates (str): Name of graph property of atomic coordinates. Default is "node_coordinates".
            atomic_symbol (str): Name of graph property of atomic symbol. Default is "node_symbol".
            atomic_number (str): Name of graph property of atomic number. Default is "node_number".

        Returns:
            self
        """
        xyz_list = self.get_geom_from_xyz_file(file_path)
        symbol = [np.array(x[0]) for x in xyz_list]
        coord = [np.array(x[1], dtype="float")[:, :3] for x in xyz_list]
        nodes = [np.array([self._global_proton_dict[x] for x in y[0]], dtype="int") for y in xyz_list]
        for key, value in zip([atomic_coordinates, atomic_symbol, atomic_number], [coord, symbol, nodes]):
            if key is not None:
                self.assign_property(key, value)
        return self

    def set_attributes(self,
                       label_column_name: Union[str, list] = None,
                       nodes: list = None,
                       edges: list = None,
                       graph: list = None,
                       encoder_nodes: dict = None,
                       encoder_edges: dict = None,
                       encoder_graph: dict = None,
                       add_hydrogen: bool = True,
                       make_directed: bool = False,
                       sanitize: bool = False,
                       compute_partial_charges: str = None,
                       additional_callbacks: Dict[str, Callable[[MolGraphInterface, dict], None]] = None,
                       custom_transform: Callable[[MolGraphInterface], MolGraphInterface] = None
                       ):
        """Read SDF-file with chemical structure information into memory.

        Args:
            label_column_name (str, list): Name of labels for columns in CSV file.
            nodes (list): A list of node attributes as string. In place of names also functions can be added.
            edges (list): A list of edge attributes as string. In place of names also functions can be added.
            graph (list): A list of graph attributes as string. In place of names also functions can be added.
            encoder_nodes (dict): A dictionary of callable encoder where the key matches the attribute.
            encoder_edges (dict): A dictionary of callable encoder where the key matches the attribute.
            encoder_graph (dict): A dictionary of callable encoder where the key matches the attribute.
            add_hydrogen (bool): Whether to keep hydrogen after reading the mol-information. Default is False.
            make_directed (bool): Whether to have directed or undirected bonds. Default is False.
            sanitize (bool): Whether to sanitize molecule. Default is False.
            compute_partial_charges (str): Whether to compute partial charges, e.g. 'gasteiger'. Default is None.
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
        additional_callbacks = additional_callbacks if additional_callbacks is not None else {}

        # Deserializing encoders.
        for encoder in [encoder_nodes, encoder_edges, encoder_graph]:
            if encoder is not None:
                for key, value in encoder.items():
                    encoder[key] = deserialize_encoder(value)

        callbacks = {
            "node_symbol": lambda mg, ds: mg.node_symbol,
            "node_number": lambda mg, ds: mg.node_number,
            "node_coordinates": lambda mg, ds: mg.node_coordinates,
            "edge_indices": lambda mg, ds: mg.edge_number[0],
            "edge_number": lambda mg, ds: np.array(mg.edge_number[1], dtype='int'),
            **additional_callbacks
        }
        # Label callback.
        if label_column_name:
            callbacks.update({'graph_labels': lambda mg, ds: ds[label_column_name]})

        # Attributes callbacks.
        if nodes:
            callbacks.update({
                'node_attributes': lambda mg, ds: np.array(mg.node_attributes(nodes, encoder_nodes), dtype='float32')
            })
        if edges:
            callbacks.update({
                'edge_attributes': lambda mg, ds: np.array(mg.edge_attributes(edges, encoder_edges)[1], dtype='float32')
            })
        if graph:
            callbacks.update({
                'graph_attributes': lambda mg, ds: np.array(mg.graph_attributes(graph, encoder_graph), dtype='float32')
            })

        value_list = map_molecule_callbacks(
            self.get_mol_blocks_from_sdf_file(),
            self.read_in_table_file().data_frame,
            callbacks=callbacks,
            add_hydrogen=add_hydrogen,
            custom_transform=custom_transform,
            make_directed=make_directed,
            sanitize=sanitize,
            mol_interface_class=self._mol_graph_interface,
            logger=self.logger,
            loop_update_info=self._default_loop_update_info,
            compute_partial_charges=compute_partial_charges
        )

        for name, values in value_list.items():
            self.assign_property(name, values)

        return self

    def read_in_memory(self,
                       label_column_name: Union[str, list] = None,
                       nodes: list = None,
                       edges: list = None,
                       graph: list = None,
                       encoder_nodes: dict = None,
                       encoder_edges: dict = None,
                       encoder_graph: dict = None,
                       add_hydrogen: bool = True,
                       sanitize: bool = False,
                       make_directed: bool = False,
                       compute_partial_charges: bool = False,
                       additional_callbacks: Dict[str, Callable[[MolGraphInterface, dict], None]] = None,
                       custom_transform: Callable[[MolGraphInterface], MolGraphInterface] = None):
        """Read geometric information into memory.

        Graph labels require a column specified by :obj:`label_column_name`.

        Args:
            label_column_name (str, list): Name of labels for columns in CSV file.
            nodes (list): A list of node attributes as string. In place of names also functions can be added.
            edges (list): A list of edge attributes as string. In place of names also functions can be added.
            graph (list): A list of graph attributes as string. In place of names also functions can be added.
            encoder_nodes (dict): A dictionary of callable encoder where the key matches the attribute.
            encoder_edges (dict): A dictionary of callable encoder where the key matches the attribute.
            encoder_graph (dict): A dictionary of callable encoder where the key matches the attribute.
            add_hydrogen (bool): Whether to keep hydrogen after reading the mol-information. Default is False.
            make_directed (bool): Whether to have directed or undirected bonds. Default is False.
            compute_partial_charges (str): Whether to compute partial charges, e.g. 'gasteiger'. Default is None.
            sanitize (bool): Whether to sanitize molecule. Default is False.
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
        if os.path.exists(self.file_path_mol) and self._mol_graph_interface is not None:
            self.info("Reading structures from SDF file.")
            self.set_attributes(
                label_column_name=label_column_name, nodes=nodes, edges=edges, graph=graph, encoder_nodes=encoder_nodes,
                encoder_edges=encoder_edges, encoder_graph=encoder_graph, add_hydrogen=add_hydrogen,
                make_directed=make_directed, additional_callbacks=additional_callbacks, sanitize=sanitize,
                custom_transform=custom_transform
            )
        else:
            # Try to read labels and xyz-file without mol-interface.
            self.warning("Failed to load structures SDF file. Reading geometries from XYZ file instead. Please check.")
            data_frame = self.read_in_table_file().data_frame
            if data_frame is not None and label_column_name is not None:
                self.assign_property("graph_labels", [
                    np.array(data_frame.loc[index][label_column_name]) for index in range(data_frame.shape[0])])
            self.read_in_memory_xyz()
        return self

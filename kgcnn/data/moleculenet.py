import os
import numpy as np
import pandas as pd

from typing import Dict, Callable, Union, List
from collections import defaultdict
from kgcnn.molecule.serial import deserialize_encoder
from kgcnn.data.base import MemoryGraphDataset
from kgcnn.molecule.base import MolGraphInterface
from kgcnn.molecule.encoder import OneHotEncoder
from kgcnn.molecule.io import write_mol_block_list_to_sdf, read_mol_list_from_sdf_file, write_smiles_file
from kgcnn.molecule.convert import MolConverter

try:
    from kgcnn.molecule.graph_rdkit import MolecularGraphRDKit
except ModuleNotFoundError:
    MolecularGraphRDKit = None


def map_molecule_callbacks(mol_list: List[str],
                           data: Union[pd.Series, pd.DataFrame],
                           callbacks: Dict[str, Callable[[MolGraphInterface, pd.Series], None]],
                           custom_transform: Callable[[MolGraphInterface], MolGraphInterface] = None,
                           add_hydrogen: bool = False,
                           make_directed: bool = False,
                           sanitize: bool = True,
                           compute_partial_charges: str = None,
                           mol_interface_class=None,
                           logger=None,
                           loop_update_info: int = 5000
                           ) -> dict:
    r"""This method receive the list of molecules, as well as the data from a pandas data series.
    It then iterates over all the molecules / data rows and invokes the callbacks for each.

    The "callbacks" parameter is supposed to be a dictionary whose keys are string names of attributes which are
    supposed to be derived from the molecule / data and the values are function objects which define how to
    derive that data. Those callback functions get two parameters:

        - mg: The :obj:`MolGraphInterface` instance for the current molecule
        - ds: A pandas data series that match data in the CSV file for the specific molecule.

    The string keys of the "callbacks" directory are also the string names which are later used to assign the
    properties of the underlying :obj:`GraphList`. This means that each element of the dataset will then have a
    field with the same name.

    .. note::

        If a molecule cannot be properly loaded by :obj:`MolGraphInterface`, then for all attributes
        "None" is added without invoking the callback!

    Example:

    .. code-block:: python

        mol_net = MoleculeNetDataset()
        mol_net.prepare_data()

        mol_values = map_molecule_callbacks(
            mol_net.get_mol_blocks_from_sdf_file(),
            mol_net.read_in_table_file().data_frame,
            callbacks={
                'graph_size': lambda mg, dd: len(mg.node_number),
                'index': lambda mg, dd: dd['index']
            }
        )

        for key, value in mol_values.items():
            mol_net.assign_property(key, value)

        mol: dict = mol_net[0]
        assert 'graph_size' in mol.keys()
        assert 'index' in mol.keys()


    Args:
        mol_list (list): List of mol strings.
        data (pd.DataFrame): Pandas data frame or series matching list of mol-strings.
        callbacks (dict): Dictionary of callbacks to perform on MolecularGraph object and table entries.
        add_hydrogen (bool): Whether to add hydrogen when making a :obj:`MolecularGraphRDKit` instance.
        make_directed (bool): Whether to have directed or undirected bonds. Default is False.
        sanitize (bool): Whether to sanitize molecule. Default is True.
        custom_transform (Callable): Custom transformation function to modify the generated
            :obj:`MolecularGraphRDKit` before callbacks are carried out. The function must take a single
            :obj:`MolecularGraphRDKit` instance as argument and return a (new) :obj:`MolecularGraphRDKit` instance.
        compute_partial_charges (str): Whether to compute partial charges, e.g. 'gasteiger'. Default is None.
        mol_interface_class: Interface for molecular graphs. Must be a :obj:`MolGraphInterface`.
        logger: Logger to report error and progress.
        loop_update_info (int): Updates for processed molecules.

    Returns:
        dict: Values of callbacks.
    """
    # Dictionaries values are lists, one for each attribute defines in "callbacks" and each value in those
    # lists corresponds to one molecule in the dataset.
    if data is None:
        if logger is not None:
            logger.error("Received no pandas data.")
    if mol_list is None:
        raise ValueError("Expected list of mol-string. But got '%s'." % mol_list)

    value_lists = defaultdict(list)
    for index, sm in enumerate(mol_list):

        mg = mol_interface_class(make_directed=make_directed).from_mol_block(
            sm, keep_hs=add_hydrogen, sanitize=sanitize)

        if custom_transform is not None:
            mg = custom_transform(mg)

        if compute_partial_charges:
            mg.compute_partial_charges(method=compute_partial_charges)

        for name, callback in callbacks.items():
            if mg.mol is None:
                value_lists[name].append(None)
            else:
                if data is not None:
                    data_dict = data.loc[index]
                else:
                    data_dict = None
                value = callback(mg, data_dict)
                value_lists[name].append(value)
        if index % loop_update_info == 0:
            if logger is not None:
                logger.info(" ... process molecules {0} from {1}".format(index, len(mol_list)))

    return value_lists


class MoleculeNetDataset(MemoryGraphDataset):
    r"""Class for using 'MoleculeNet' datasets.

    The concept is to load a table of smiles and corresponding targets and convert them into a tensor representation
    for graph networks.

    .. code-block:: console

        ├── data_directory
            ├── file_name.csv
            ├── file_name.SMILES
            ├── file_name.sdf
            └── dataset_name.kgcnn.pickle

    The class provides properties and methods for making graph features from smiles.
    The typical input is a `csv` or `excel` file with smiles and corresponding graph labels.
    The table file must have one line of header with column names!

    The graph structure matches the molecular graph, i.e. the chemical structure. The atomic coordinates
    are generated by a conformer guess. Since this require some computation time, it is only done once and the
    molecular coordinate or mol-blocks stored in a single SDF file with the base-name of the csv :obj:``file_name``.
    Conversion is using the :obj:`MolConverter` class.

    The selection of smiles and whether conformers should be generated is handled by subclasses or specified in
    the methods :obj:`prepare_data` and :obj:`read_in_memory`, see the documentation of the methods
    for further details.

    Attribute generation is carried out via the :obj:`MolecularGraphRDKit` class and requires RDKit as backend.
    You can also use a pre-processed SDF or SMILES file in :obj:`data_directory` and add their name in the
    class initialization.
    """

    _default_node_attributes = [
        'Symbol', 'TotalDegree', 'FormalCharge', 'NumRadicalElectrons', 'Hybridization',
        'IsAromatic', 'IsInRing', 'TotalNumHs', 'CIPCode', "ChiralityPossible", "ChiralTag"
    ]
    _default_node_encoders = {
        'Symbol': OneHotEncoder(
            ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
            dtype="str"
        ),
        'Hybridization': OneHotEncoder([2, 3, 4, 5, 6]),
        'TotalDegree': OneHotEncoder([0, 1, 2, 3, 4, 5], add_unknown=False),
        'TotalNumHs': OneHotEncoder([0, 1, 2, 3, 4], add_unknown=False),
        'CIPCode': OneHotEncoder(['R', 'S'], add_unknown=False, dtype='str'),
        "ChiralityPossible": OneHotEncoder(["1"], add_unknown=False, dtype='str'),
    }
    _default_edge_attributes = ['BondType', 'IsAromatic', 'IsConjugated', 'IsInRing', 'Stereo']
    _default_edge_encoders = {
        'BondType': OneHotEncoder([1, 2, 3, 12], add_unknown=False),
        'Stereo': OneHotEncoder([0, 1, 2, 3], add_unknown=False)
    }
    _default_graph_attributes = ['ExactMolWt', 'NumAtoms']
    _default_graph_encoders = {}

    _default_loop_update_info = 5000
    _mol_graph_interface = MolecularGraphRDKit

    def __init__(self, data_directory: str = None, dataset_name: str = None, file_name: str = None,
                 file_name_mol: str = None, file_name_smiles: str = None, verbose: int = 10):
        r"""Initialize a :obj:`MoleculeNetDataset` with information of the dataset location on disk.

        Args:
            file_name (str): Filename for reading into memory. This must be the name of the '.csv' file.
                Default is None.
            file_name_mol (str): Filename of the SDF file that is generated from the SMILES file that is generated
                from a list of smiles given in the table file specified by :obj:`file_name` . By default, the name
                is chosen equal to :obj:`file_name` when passed None.
            file_name_smiles (str): Filename of the SMILES file that is generated from a list of smiles given in the
                table file specified by :obj:`file_name` . By default, the name is chosen equal to :obj:`file_name`
                when passed None.
            data_directory (str): Full path to directory containing all dataset files. Default is None.
                Not used by this subclass. Ignored.
            dataset_name (str): Name of the dataset. Important for naming. Default is None.
            verbose (int): Logging level. Default is 10.
        """
        MemoryGraphDataset.__init__(self, data_directory=data_directory, dataset_name=dataset_name,
                                    file_name=file_name, verbose=verbose)
        self.file_name_mol = file_name_mol
        self.file_name_smiles = file_name_smiles

    @property
    def file_path_mol(self):
        """Try to determine a file path for the mol information to store."""
        self._verify_data_directory()
        if self.file_name_mol is None:
            return os.path.join(self.data_directory, os.path.splitext(self.file_name)[0] + ".sdf")
        else:
            return os.path.join(self.data_directory, self.file_name_mol)

    @property
    def file_path_smiles(self):
        """Try to determine a file path for the SMILES information to store."""
        self._verify_data_directory()
        if self.file_name_smiles is None:
            return os.path.join(self.data_directory, os.path.splitext(self.file_name)[0] + ".SMILES")
        else:
            return os.path.join(self.data_directory, self.file_name_smiles)

    def prepare_data(self, overwrite: bool = False, smiles_column_name: str = "smiles",
                     add_hydrogen: bool = True, sanitize: bool = True,
                     make_conformers: bool = True, optimize_conformer: bool = True,
                     external_program: dict = None, num_workers: int = None):
        r"""Computation of molecular structure information and optionally conformers from smiles.

        This function reads smiles from the csv-file given by :obj:`file_name` and creates a single SDF File of
        generated mol-blocks with the same file name.
        The function requires :obj:`RDKit` and (optionally) :obj:`OpenBabel`.
        Smiles that are not compatible with both RDKit and OpenBabel result in an empty mol-block in the SDF file to
        keep the number of molecules the same.

        Args:
            overwrite (bool): Overwrite existing database mol-json file. Default is False.
            smiles_column_name (str): Column name where smiles are given in csv-file. Default is "smiles".
            add_hydrogen (bool): Whether to add H after smile translation. Default is True.
            sanitize (bool): Whether to sanitize molecule. Default is True.
            make_conformers (bool): Whether to make conformers. Default is True.
            optimize_conformer (bool): Whether to optimize conformer via force field.
                Only possible with :obj:`make_conformers`. Default is True.
            external_program (dict): External program for translating smiles. Default is None.
                If you want to use an external program you have to supply a dictionary of the form:
                {"class_name": "balloon", "config": {"balloon_executable_path": ..., ...}}.
                Note that usually the parameters like :obj:`add_hydrogen` are ignored. And you need to control the
                SDF file generation within `config` of the :obj:`external_program`.
            num_workers (int): Parallel execution for translating smiles.

        Returns:
            self
        """
        if os.path.exists(self.file_path_mol) and not overwrite:
            self.info("Found SDF %s of pre-computed structures." % self.file_path_mol)
            return self

        self.read_in_table_file()
        smiles = self.data_frame[smiles_column_name].values
        if len(smiles) == 0:
            self.error("Can not translate smiles, received empty list for '%s'." % self.dataset_name)
        write_smiles_file(self.file_path_smiles, smiles)

        # Make structure
        self.info("Generating molecules and store %s to disk..." % self.file_path_mol)
        conv = MolConverter()
        conv.smile_to_mol(
            self.file_path_smiles, self.file_path_mol, add_hydrogen=add_hydrogen, sanitize=sanitize,
            make_conformers=make_conformers, optimize_conformer=optimize_conformer,
            external_program=external_program, num_workers=num_workers,
            logger=self.logger, batch_size=self._default_loop_update_info
        )
        return self

    def get_mol_blocks_from_sdf_file(self):
        if not os.path.exists(self.file_path_mol):
            raise FileNotFoundError("Can not load molecules for dataset %s" % self.dataset_name)

        # Loading the molecules and the csv data
        self.info("Read molecules from mol-file.")
        return read_mol_list_from_sdf_file(self.file_path_mol)

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
                       sanitize: bool = True,
                       compute_partial_charges: str = None,
                       additional_callbacks: Dict[str, Callable[[MolGraphInterface, dict], None]] = None,
                       custom_transform: Callable[[MolGraphInterface], MolGraphInterface] = None):
        """Load list of molecules from cached SDF-file in into memory. File name must be given in :obj:`file_name` and
        path information in the constructor of this class.

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

            * mg: The :obj:`MolecularGraphRDKit` instance of the molecule
            * ds: A pandas data series that match data in the CSV file for the specific molecule.

        Example:

        .. code-block:: python

            from os import linesep
            csv = f"index,name,label,smiles{linesep}1,Propanolol,1,[Cl].CC(C)NCC(O)COc1cccc2ccccc12"
            with open('/tmp/moleculenet_example.csv', mode='w') as file:
                file.write(csv)

            dataset = MoleculeNetDataset('/tmp', 'example', 'moleculenet_example.csv')
            dataset.prepare_data(smiles_column_name='smiles')
            dataset.read_in_memory(label_column_name='label')
            dataset.set_attributes(
                nodes=['Symbol'],
                encoder_nodes={'Symbol': OneHotEncoder(['C', 'O'], dtype='str')},
                edges=['BondType'],
                encoder_edges={'BondType': int},
                additional_callbacks={
                    # It is important that the callbacks return a numpy array, even if it is just a single element.
                    'name': lambda mg, ds: np.array(ds['name'], dtype='str')
                }
            )

            mol: dict = dataset[0]
            mol['node_attributes']  # np array of one hot encoded atom type per node
            mol['edge_attributes']  # int value representing the bond type
            mol['name']  # Array of a single string which is the name from the original CSV data

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
            sanitize (bool): Whether to sanitize molecule. Default is True.
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
                encoder[key] = deserialize_encoder(value)

        callbacks = {
            'node_symbol': lambda mg, ds: mg.node_symbol,
            'node_number': lambda mg, ds: mg.node_number,
            'edge_indices': lambda mg, ds: mg.edge_number[0],
            'edge_number': lambda mg, ds: np.array(mg.edge_number[1], dtype='int'),
            'graph_size': lambda mg, ds: len(mg.node_number),
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

        value_lists = map_molecule_callbacks(
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

        for name, values in value_lists.items():
            self.assign_property(name, values)

        if self.logger.getEffectiveLevel() < 20:
            for encoder in [encoder_nodes, encoder_edges, encoder_graph]:
                for key, value in encoder.items():
                    if hasattr(value, "report"):
                        value.report(name=key)

        return self

    read_in_memory = set_attributes

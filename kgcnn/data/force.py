import os
import numpy as np
from kgcnn.molecule.base import MolGraphInterface
from typing import Union, Callable, Dict
from kgcnn.data.qm import QMDataset
from kgcnn.molecule.io import parse_list_to_xyz_str, read_xyz_file, write_list_to_xyz_file


class ForceDataset(QMDataset):
    r"""This is a base class for Force datasets. Inherits all functionality from QMDataset.

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
            ├── file_name_force.xyz
            ├── ...
            └── dataset_name.kgcnn.pickle

    Additionally, forces xyz information can be read in with this class.
    """

    def __init__(self, data_directory: str = None, dataset_name: str = None, file_name: str = None,
                 verbose: int = 10, file_directory: str = None,
                 file_name_xyz: str = None,
                 file_name_mol: str = None,
                 file_name_force_xyz: str = None):
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
        super(ForceDataset, self).__init__(data_directory=data_directory, dataset_name=dataset_name,
                                           file_name=file_name, verbose=verbose, file_directory=file_directory)
        self.label_units = None
        self.label_names = None
        self.file_name_xyz = file_name_xyz
        self.file_name_mol = file_name_mol
        self.file_name_force_xyz = file_name_force_xyz

    @property
    def file_path_force_xyz(self):
        """Try to determine a file name for the mol information to store."""
        self._verify_data_directory()
        if self.file_name_force_xyz is None:
            return os.path.join(self.data_directory, os.path.splitext(self.file_name)[0] + "_force.xyz")
        elif isinstance(self.file_name_force_xyz, (str, os.PathLike)):
            return os.path.join(self.data_directory, self.file_name_force_xyz)
        elif isinstance(self.file_name_force_xyz, (list, tuple)):
            return [os.path.join(self.data_directory, x) for x in self.file_name_force_xyz]
        else:
            raise TypeError("Wrong type for `file_name_force_xyz` : '%s'." % self.file_name_force_xyz)

    def prepare_data(self, overwrite: bool = False, file_column_name: str = None, file_column_name_force: str = None,
                     make_sdf: bool = False):
        r"""Pre-computation of molecular structure information in a sdf-file from a xyz-file or a folder of xyz-files.

        If there is no single xyz-file, it will be created with the information of a csv-file with the same name.

        Args:
            overwrite (bool): Overwrite existing database SDF file. Default is False.
            file_column_name (str): Name of the column in csv file with list of xyz-files located in file_directory.
                This is for the positions only.
            file_column_name_force (str, list): Column name of xyz files for forces in file directory.
            make_sdf (bool): Whether to try to make a sdf file from xyz information via `RDKit` and `OpenBabel`.

        Returns:
            self
        """
        super(ForceDataset, self).prepare_data(overwrite=overwrite, file_column_name=file_column_name,
                                               make_sdf=make_sdf)

        # Try collect single xyz files in directory
        file_path_forces = self.file_path_force_xyz
        if not isinstance(file_path_forces, (list, tuple)):
            file_path_forces = [file_path_forces]
        if not isinstance(file_column_name_force, (list, tuple)):
            file_column_name_force = [file_column_name_force]
        for f, c in zip(file_path_forces, file_column_name_force):
            if not os.path.exists(f):
                xyz_list = self.collect_files_in_file_directory(
                    file_column_name=c, table_file_path=None,
                    read_method_file=self.get_geom_from_xyz_file, update_counter=self._default_loop_update_info,
                    append_file_content=True, read_method_return_list=True
                )
                write_list_to_xyz_file(f, xyz_list)

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
        super(ForceDataset, self).read_in_memory(
            label_column_name=label_column_name, nodes=nodes, edges=edges, graph=graph, encoder_nodes=encoder_nodes,
            encoder_edges=encoder_edges, encoder_graph=encoder_graph, add_hydrogen=add_hydrogen, sanitize=sanitize,
            make_directed=make_directed, compute_partial_charges=compute_partial_charges,
            additional_callbacks=additional_callbacks, custom_transform=custom_transform
        )
        file_path_forces = self.file_path_force_xyz
        if not isinstance(file_path_forces, (list, tuple)):
            file_path_forces = [file_path_forces]
        for i, x in enumerate(file_path_forces):
            self.read_in_memory_xyz(x, atomic_coordinates=str(os.path.basename(x)),
                                    # We do not want to change atomic information but only read xyz.
                                    atomic_number=None, atomic_symbol=None)
        return self

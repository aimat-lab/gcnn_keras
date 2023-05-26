import os
import numpy as np
from collections import defaultdict
from typing import Dict, Callable, List, Union
import pandas as pd
import pymatgen
import pymatgen.io.cif
import pymatgen.core.structure
import pymatgen.symmetry.structure
from kgcnn.utils.serial import deserialize
from kgcnn.data.base import MemoryGraphDataset
from kgcnn.data.utils import save_json_file, load_json_file
from kgcnn.crystal.base import CrystalPreprocessor
from kgcnn.graph.base import GraphDict


class CrystalDataset(MemoryGraphDataset):
    r"""Class for making graph dataset from periodic structures such as crystals.

    The dataset class requires a :obj:`data_directory` to store a table '.csv' file containing labels and information
    of the structures stored in multiple (CIF, POSCAR, ...) files in :obj:`file_directory` .
    The file names must be included in the '.csv' table. The table file must have one line of header with column names!

    .. code-block:: console

        ├── data_directory
            ├── file_directory
            │   ├── *.cif
            │   ├── *.cif
            │   └── ...
            ├── file_name.csv
            ├── file_name.pymatgen.json
            └── dataset_name.kgcnn.pickle

    This class uses :obj:`pymatgen.core.structure.Structure` and therefore requires :obj:`pymatgen` to be installed.
    A '.pymatgen.json' serialized file is generated to store a list of structures from single '.cif' files via
    :obj:`prepare_data()` .
    Consequently, a 'file_name.pymatgen.json' can be directly stored in :obj:`data_directory`.
    In this, case :obj:`prepare_data()` does not have to be used. Additionally, a table file 'file_name.csv'
    that lists the single file names and possible labels or classification targets is required.

    .. code-block:: python

        from kgcnn.data.crystal import CrystalDataset
        dataset = CrystalDataset(
            data_directory="data_directory/",
            dataset_name="ExampleCrystal",
            file_name="file_name.csv",
            file_directory="file_directory")
        dataset.prepare_data(file_column_name="file_name", overwrite=True)
        dataset.read_in_memory(label_column_name="label")
    """

    _default_loop_update_info = 5000

    def __init__(self,
                 data_directory: str = None,
                 dataset_name: str = None,
                 file_name: str = None,
                 file_directory: str = None,
                 file_name_pymatgen_json: str = None,
                 verbose: int = 10):
        r"""Initialize a base class of :obj:`CrystalDataset`.

        Args:
            data_directory (str): Full path to directory of the dataset. Default is None.
            file_name (str): Filename for dataset to read into memory. This is a table file.
                The '.csv' should contain file names that are expected to be CIF-files in :obj:`file_directory`.
                Default is None.
            file_directory (str): Name or relative path from :obj:`data_directory` to a directory containing sorted
                'cif' files. Default is None.
            file_name_pymatgen_json (str): This class will generate a 'json' file with pymatgen structures. You
                can specify the file name of that file with this argument. By default, it will be named from
                :obj:`file_name` when passed None.
            dataset_name (str): Name of the dataset. Important for naming and saving files. Default is None.
            verbose (int): Logging level. Default is 10.
        """
        super(CrystalDataset, self).__init__(
            data_directory=data_directory, dataset_name=dataset_name, file_name=file_name, verbose=verbose,
            file_directory=file_directory)
        self._structs = None
        self.file_name_pymatgen_json = file_name_pymatgen_json
        self.label_units = None
        self.label_names = None

    @property
    def pymatgen_json_file_path(self):
        """Internal file name for the pymatgen serialization information to store to disk."""
        self._verify_data_directory()
        if self.file_name_pymatgen_json is None:
            file_name = os.path.splitext(self.file_name)[0] + ".pymatgen.json"
        else:
            file_name = self.file_name_pymatgen_json
        return os.path.join(self.data_directory, file_name)

    @staticmethod
    def _pymatgen_serialize_structs(structs: List) -> List[dict]:
        dicts = []
        for s in structs:
            d = s.as_dict()
            # Module information should be already obtained from as_dict().
            # d["@module"] = type(s).__module__
            # d["@class"] = type(s).__name__
            dicts.append(d)
        return dicts

    @staticmethod
    def _pymatgen_deserialize_dicts(dicts: List[dict], to_unit_cell: bool = False) -> list:
        structs = []
        for x in dicts:
            # TODO: We could check symmetry or @module, @class items in dict.
            s = pymatgen.core.structure.Structure.from_dict(x)
            structs.append(s)
            if to_unit_cell:
                for site in s.sites:
                    site.to_unit_cell(in_place=True)
        return structs

    def save_structures_to_json_file(self, structs: list, file_path: str = None):
        """Save a list of pymatgen structures to file.

        Args:
            structs (list): List of pymatgen structures.
            file_path (str): File path to store structures to disk, uses class-default. Default is None.

        Returns:
            None.
        """
        if file_path is None:
            file_path = self.pymatgen_json_file_path
        self.info("Exporting as dict for pymatgen ...")
        dicts = self._pymatgen_serialize_structs(structs)
        self.info("Saving structures as .json ...")
        save_json_file(dicts, file_path)

    @staticmethod
    def _pymatgen_parse_file_to_structure(cif_file: str):
        # TODO: We can add flexible parsing to include other than just CIF from file here.
        structures = pymatgen.io.cif.CifParser(cif_file).get_structures()
        return structures

    def prepare_data(self, file_column_name: str = None, overwrite: bool = False):
        r"""Default preparation for crystal datasets.

        Try to load all crystal structures from single files and save them as a pymatgen json serialization.
        Can load multiple CIF files from a table that keeps file names and possible labels or additional information.

        Args:
            file_column_name (str): Name of the column that has file names found in file_directory. Default is None.
            overwrite (bool): Whether to rerun the data extraction. Default is False.

        Returns:
            self
        """
        if os.path.exists(self.pymatgen_json_file_path) and not overwrite:
            self.info("Pickled pymatgen structures already exist. Do nothing.")
            return self

        self.info("Searching for structure files in '%s'" % self.file_directory_path)
        structs = self.collect_files_in_file_directory(
            file_column_name=file_column_name, table_file_path=None,
            read_method_file=self._pymatgen_parse_file_to_structure, update_counter=self._default_loop_update_info,
            append_file_content=True, read_method_return_list=True
        )
        self.save_structures_to_json_file(structs)
        return self

    def get_structures_from_json_file(self, file_path: str = None) -> List:
        """Load pymatgen serialized json-file into memory.

        Structures are not added to :obj:`CrystalDataset` but returned by this function.

        Args:
            file_path (str): File path to json-file, uses class default. Default is None.

        Returns:
            list: List of pymatgen structures.
        """
        if file_path is None:
            file_path = self.pymatgen_json_file_path

        if not os.path.exists(file_path):
            raise FileNotFoundError("Cannot find .json file for `CrystalDataset`. Please `prepare_data()`.")

        self.info("Reading structures from .json ...")
        return self._pymatgen_deserialize_dicts(load_json_file(file_path))

    def _map_callbacks(self, structs: list, data: pd.Series,
                       callbacks: Dict[
                           str, Callable[[pymatgen.core.structure.Structure, pd.Series], Union[np.ndarray, None]]],
                       assign_to_self: bool = True) -> dict:
        """Map callbacks on a data series object plus structure list.

        Args:
            structs (list): List of pymatgen structures.
            data (pd.Series, pd.DataFrame): Data Frame matching the structure list.
            callbacks (dict): Dictionary of callbacks that take a data object plus pymatgen structure as argument.
            assign_to_self (bool): Whether to already assign the output of callbacks to this class.

        Returns:
            dict: Values of callbacks.
        """

        # The dictionaries values are lists, one for each attribute defines in "callbacks" and each value in those
        # lists corresponds to one structure in the dataset.
        value_lists = defaultdict(list)
        for index, st in enumerate(structs):
            for name, callback in callbacks.items():
                if st is None:
                    value_lists[name].append(None)
                else:
                    data_dict = data.loc[index]
                    value = callback(st, data_dict)
                    value_lists[name].append(value)
            if index % self._default_loop_update_info == 0:
                self.info(" ... read structures {0} from {1}".format(index, len(structs)))

        # The string key names of the original "callbacks" dict are also used as the names of the properties which are
        # assigned
        if assign_to_self:
            for name, values in value_lists.items():
                self.assign_property(name, values)

        return value_lists

    def read_in_memory(self, label_column_name: str = None,
                       additional_callbacks: Dict[
                           str, Callable[[pymatgen.core.structure.Structure, pd.Series], None]] = None
                       ):
        """Read structures from pymatgen json serialization and convert them into graph information.

        Args:
            label_column_name (str): Columns of labels for graph in table file. Default is None.
            additional_callbacks (dict): Callbacks to add during read into memory.

        Returns:
            self
        """
        if additional_callbacks is None:
            additional_callbacks = {}

        self.info("Making node features from structure...")
        callbacks = {"graph_labels": lambda st, ds: ds[label_column_name] if label_column_name is not None else None,
                     "node_coordinates": lambda st, ds: np.array(st.cart_coords, dtype="float"),
                     "node_frac_coordinates": lambda st, ds: np.array(st.frac_coords, dtype="float"),
                     "graph_lattice": lambda st, ds: np.ascontiguousarray(np.array(st.lattice.matrix), dtype="float"),
                     "abc": lambda st, ds: np.array(st.lattice.abc),
                     "charge": lambda st, ds: np.array([st.charge], dtype="float"),
                     "volume": lambda st, ds: np.array([st.lattice.volume], dtype="float"),
                     "node_number": lambda st, ds: np.array(st.atomic_numbers, dtype="int"),
                     **additional_callbacks
                     }

        self._map_callbacks(structs=self.get_structures_from_json_file(),
                            data=self.read_in_table_file(file_path=self.file_path).data_frame,
                            callbacks=callbacks)

        return self

    def set_representation(self, pre_processor: Union[CrystalPreprocessor, dict], reset_graphs: bool = False):
        r"""Build a graph representation for this dataset using :obj:`kgcnn.crystal` .

        Args:
            pre_processor (CrystalPreprocessor): Crystal preprocessor to use.
            reset_graphs (bool): Whether to reset the graph information. Default is False.

        Returns:

        """
        if reset_graphs:
            self.clear()
        if isinstance(pre_processor, dict):
            pre_processor = deserialize(pre_processor)
        # Read pymatgen JSON file from file.
        structs = self.get_structures_from_json_file()
        if reset_graphs:
            self.empty(len(structs))

        pre_processor.output_graph_as_dict = True

        for index, s in enumerate(structs):
            g = pre_processor(s)
            self[index].update(g)

            if index % self._default_loop_update_info == 0:
                self.info(" ... preprocess structures {0} from {1}".format(index, len(structs)))

        return self

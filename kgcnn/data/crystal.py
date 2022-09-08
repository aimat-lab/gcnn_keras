import os
import numpy as np
from collections import defaultdict
from typing import Dict, Callable, List, Union
import pandas as pd
import pymatgen.io.cif
import pymatgen.core.structure
import pymatgen.symmetry.structure

from kgcnn.data.base import MemoryGraphDataset
from kgcnn.data.utils import save_json_file, load_json_file
from kgcnn.crystal.base import CrystalPreprocessor
from kgcnn.graph.base import GraphDict


class CrystalDataset(MemoryGraphDataset):
    r"""Class for making graph dataset from periodic structures such as crystals.

    The dataset class requires a :obj:`data_directory` to store a table '.csv' file containing labels and information
    of the structures stored in either a single '.cif' file or multiple CIF-files in :obj:`file_directory`.
    In the latter case, the file names must be included in the '.csv' table.

    .. code-block:: type

        ├── data_directory
            ├── file_directory
            │   ├── *.cif
            │   ├── *.cif
            │   └── ...
            ├── file_name.csv
            └── file_name.pymatgen.json

    This class uses :obj:`pymatgen.core.structure.Structure` and therefore requires :obj:`pymatgen` to be installed.
    A '.pymatgen.json' serialized file is generated to store a list of structures from '.cif' file format via
    :obj:`prepare_data()`.

    """

    DEFAULT_LOOP_UPDATE_INFO = 5000

    def __init__(self,
                 data_directory: str = None,
                 dataset_name: str = None,
                 file_name: str = None,
                 file_directory: str = None,
                 verbose: int = 10):
        r"""Initialize a base class of :obj:`CrystalDataset`.

        Args:
            data_directory (str): Full path to directory of the dataset. Default is None.
            file_name (str): Filename for dataset to read into memory. This can be a single a 'cif' file.
                Or a '.csv' of file names that are expected to be cif-files in file_directory.
                Default is None.
            file_directory (str): Name or relative path from :obj:`data_directory` to a directory containing sorted
                'cif' files. Default is None.
            dataset_name (str): Name of the dataset. Important for naming and saving files. Default is None.
            verbose (int): Logging level. Default is 10.
        """
        super(CrystalDataset, self).__init__(data_directory=data_directory, dataset_name=dataset_name,
                                             file_name=file_name, verbose=verbose,
                                             file_directory=file_directory)
        self._structs = None

    def _get_pymatgen_file_name(self):
        """Try to determine a file name for the pymatgen serialization information to store to disk."""
        return os.path.splitext(self.file_name)[0] + ".pymatgen.json"

    @staticmethod
    def _pymatgen_serialize_structs(structs: List):
        dicts = []
        for s in structs:
            d = s.as_dict()
            # Module information should be already obtained from as_dict().
            # d["@module"] = type(s).__module__
            # d["@class"] = type(s).__name__
            dicts.append(d)
        return dicts

    @staticmethod
    def _pymatgen_parse_cif_file_to_structures(cif_file: str):
        # structure = pymatgen.io.cif.CifParser.from_string(cif_string).get_structures()[0]
        structures = pymatgen.io.cif.CifParser(cif_file).get_structures()
        return structures

    def prepare_data(self, cif_column_name: str = None, overwrite: bool = False):
        r"""Try to load all crystal structures from CIF files and save them as a pymatgen json serialization.
        Can load a single CIF file with multiple structures (maybe unstable), or multiple CIF files from a table
        that keeps file names and possible labels or additional information.

        Args:
            cif_column_name (str): Name of the column that has file names found in file_directory. Default is None.
            overwrite (bool): Whether to rerun the data extraction. Default is False.

        Returns:
            self
        """
        if os.path.exists(os.path.join(self.data_directory, self._get_pymatgen_file_name())) and not overwrite:
            self.info("Pickled pymatgen structures already exist. Do nothing.")
            return self
        pymatgen_file_made = False

        file_path = os.path.join(self.data_directory, self.file_name)
        file_path_base = os.path.splitext(file_path)[0]

        # Check for a single CIF file.
        found_cif_file = False
        if os.path.exists(file_path_base + ".cif"):
            found_cif_file = True
            self.info("Start to read many structures form cif-file via pymatgen ...")
            structs = self._pymatgen_parse_cif_file_to_structures(file_path)
            self.info("Exporting as dict for pymatgen ...")
            dicts = self._pymatgen_serialize_structs(structs)
            self.info("Saving structures as .json ...")
            out_path = os.path.join(self.data_directory, self._get_pymatgen_file_name())
            save_json_file(dicts, out_path)
            pymatgen_file_made = True

        # We try to read in a csv file.
        self.read_in_table_file(file_path=file_path)

        # Check if table has a list of single cif files in file directory.
        if not found_cif_file and cif_column_name is not None and self.data_frame is not None:
            # Try to find file names in data_frame
            cif_file_list = self.data_frame[cif_column_name].values
            num_structs = len(cif_file_list)
            structs = []
            self.info("Read %s cif-file via pymatgen ..." % num_structs)
            for i, x in enumerate(cif_file_list):
                # Only one file per path
                structs.append(self._pymatgen_parse_cif_file_to_structures(os.path.join(self.data_directory,
                                                                                        self.file_directory, x))[0])
                if i % self.DEFAULT_LOOP_UPDATE_INFO == 0:
                    self.info(" ... read structure {0} from {1}".format(i, num_structs))
            self.info("Exporting as dict for pymatgen ...")
            dicts = self._pymatgen_serialize_structs(structs)
            self.info("Saving structures as .json ...")
            out_path = os.path.join(self.data_directory, self._get_pymatgen_file_name())
            save_json_file(dicts, out_path)
            pymatgen_file_made = True

        if not pymatgen_file_made:
            raise FileNotFoundError("Could not make pymatgen structures.")

        return self

    @staticmethod
    def _pymatgen_deserialize_dicts(dicts: List[dict], to_unit_cell: bool = False) -> list:
        structs = []
        for x in dicts:
            # We could check symmetry or @module, @class items in dict.
            s = pymatgen.core.structure.Structure.from_dict(x)
            structs.append(s)
            if to_unit_cell:
                for site in s.sites:
                    site.to_unit_cell(in_place=True)
        return structs

    def _read_pymatgen_json_in_memory(self):
        file_path = os.path.join(self.data_directory, self._get_pymatgen_file_name())

        if not os.path.exists(file_path):
            raise FileNotFoundError("Cannot find .json file for `CrystalDataset`. Please `prepare_data()`.")

        self.info("Reading structures from .json ...")

        dicts = load_json_file(file_path)
        return self._pymatgen_deserialize_dicts(dicts)

    def _map_callbacks(self,
                       structs: list,
                       data: pd.Series,
                       callbacks: Dict[
                           str, Callable[[pymatgen.core.structure.Structure, pd.Series], Union[np.ndarray, None]]],
                       assign_to_self: bool = True) -> dict:

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
            if index % self.DEFAULT_LOOP_UPDATE_INFO == 0:
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
            label_column_name (str): Columns of labels for graph. Default is None.
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

        self._map_callbacks(structs=self._read_pymatgen_json_in_memory(),
                            data=self.read_in_table_file(file_path=self.file_path).data_frame,
                            callbacks=callbacks)

        return self

    def set_representation(self, pre_processor: CrystalPreprocessor, reset_graphs: bool = False):

        if reset_graphs:
            self.clear()
        # Read pymatgen JSON file from file.
        structs = self._read_pymatgen_json_in_memory()
        if reset_graphs:
            self.empty(len(structs))

        for index, s in enumerate(structs):
            g = pre_processor(s)
            graph_dict = GraphDict()
            graph_dict.from_networkx(g, node_attributes=pre_processor.node_attributes,
                                     edge_attributes=pre_processor.edge_attributes)
            # TODO: Add graph attributes (label, lattice_matrix, etc.)
            # TODO: Rename node and edge properties to match kgcnn conventions
            # TODO: Add GraphDict to dataset

            if index % self.DEFAULT_LOOP_UPDATE_INFO == 0:
                self.info(" ... preprocess structures {0} from {1}".format(index, len(structs)))

        return self

import os
import logging
import numpy as np

try:
    import pymatgen.core.structure
except ModuleNotFoundError:
    print("Can not find `pymatgen`, but required for this module. Please install `pymatgen`!")

from kgcnn.data.base import MemoryGraphDataset
from kgcnn.utils.data import save_json_file, load_json_file, pandas_data_frame_columns_to_numpy
from kgcnn.mol.graphPyMat import parse_cif_file_to_structures, structure_get_properties, \
    structure_get_range_neighbors


class CrystalDataset(MemoryGraphDataset):
    """Class for making graph properties from periodic structures such as crystals.

    .. warning::
        Currently under construction. Not working at the moment.

    """

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

    def prepare_data(self, cif_column_name: str = None, overwrite: bool = False):
        r"""Try to load all crystal structures from CIF files and save them as a pymatgen json serialization.
        Can load single CIF file with multiple structures (maybe unstable), or multiple CIF files from a table
        that keeps file names and possible values.

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
            structs = parse_cif_file_to_structures(file_path)
            self.info("Exporting as dict for pymatgen ...")
            dicts = [s.as_dict() for s in structs]
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
                structs.append(parse_cif_file_to_structures(os.path.join(self.data_directory,
                                                                         self.file_directory, x))[0])
                if i % 1000 == 0:
                    self.info(" ... read structure {0} from {1}".format(i, num_structs))
            self.info("Exporting as dict for pymatgen ...")
            dicts = [s.as_dict() for s in structs]
            self.info("Saving structures as .json ...")
            out_path = os.path.join(self.data_directory, self._get_pymatgen_file_name())
            save_json_file(dicts, out_path)
            pymatgen_file_made = True

        if not pymatgen_file_made:
            raise FileNotFoundError("Could not make pymatgen structures.")

        return self

    def _read_pymatgen_json_in_memory(self):
        file_path = os.path.join(self.data_directory, self._get_pymatgen_file_name())
        if not os.path.exists(file_path):
            raise FileNotFoundError("Cannot find .json file for `CrystalDataset`. Please prepare_data().")

        self.info("Reading structures from .json ...")
        dicts = load_json_file(file_path)
        structs = [pymatgen.core.structure.Structure.from_dict(x) for x in dicts]
        return structs

    def read_in_memory(self, label_column_name: str = None):
        """Read structures from pymatgen json serialization and convert them into graph information.

        Args:
            label_column_name (str): Columns of labels for graph. Default is None.

        Returns:
            self
        """
        file_path = os.path.join(self.data_directory, self.file_name)

        # Try to read table file
        self.read_in_table_file(file_path=file_path)

        # We can try to get labels here.
        if self.data_frame is not None and label_column_name is not None:
            self.graph_labels = pandas_data_frame_columns_to_numpy(self.data_frame, label_column_name)

        # Read pymatgen JSON file from file.
        structs = self._read_pymatgen_json_in_memory()
        self.info("Making node features ...")
        node_symbol = []
        node_coordinates = []
        graph_lattice_matrix = []
        graph_abc = []
        graph_charge = []
        graph_volume = []
        node_occ = []
        node_oxidation = []
        for x in structs:
            coords, lat_matrix, abc, tot_chg, volume, occ, oxi, symb = structure_get_properties(x)
            node_coordinates.append(coords)
            graph_lattice_matrix.append(lat_matrix)
            graph_abc.append(abc)
            graph_charge.append(tot_chg)
            graph_volume.append(volume)
            node_symbol.append(symb)
            node_occ.append(occ)
            node_oxidation.append(oxi)

        self.node_attributes = node_occ
        self.node_oxidation = node_oxidation
        self.node_number = [np.argmax(x, axis=-1) for x in node_occ]  # Only takes maximum occupation here!!!
        self.node_symbol = node_symbol
        self.node_coordinates = node_coordinates
        self.graph_lattice_matrix = graph_lattice_matrix
        self.graph_abc = graph_abc
        self.graph_charge = graph_charge
        self.graph_volume = graph_volume

        return self

    def _set_dataset_range_from_structures(self, structs, radius: float = 4.0, numerical_tol: float = 1e-08,
                                           max_neighbours: int = 100000000):
        self.info("Setting range ...")
        range_indices = []
        range_image = []
        range_distance = []
        for x in structs:
            ridx, rimg, rd = structure_get_range_neighbors(x, radius=radius, numerical_tol=numerical_tol,
                                                           max_neighbours=max_neighbours)
            range_indices.append(ridx)
            range_image.append(rimg)
            range_distance.append(rd)
        self.range_indices = range_indices
        self.range_image = range_image
        self.range_attributes = range_distance

    def set_range(self, max_distance: float = 4.0, max_neighbours=15, do_invert_distance=False,
                  self_loops=True, exclusive=True):
        assert exclusive, "Range in `CrystalDataset` must have exclusive=True."
        structs = self._read_pymatgen_json_in_memory()
        self._set_dataset_range_from_structures(structs=structs, radius=max_distance, max_neighbours=max_neighbours)
        return self

    def set_angle(self, prefix_indices: str = "range", compute_angles: bool = False, allow_multi_edges=True):
        # Since coordinates are periodic.
        assert not compute_angles, "Can not compute angles for periodic systems."
        assert allow_multi_edges, "Require multi edges for periodic structures."
        super(CrystalDataset, self).map_list("set_angle", prefix_indices=prefix_indices, compute_angles=compute_angles,
                                             allow_multi_edges=allow_multi_edges)

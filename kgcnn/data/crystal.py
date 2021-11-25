import os

try:
    import pymatgen.core.structure
except ModuleNotFoundError:
    print("ERROR:kgcnn: Can not find `pymatgen`, but required for this module. Please install `pymatgen`!")

from kgcnn.data.base import MemoryGraphDataset
from kgcnn.utils.data import save_json_file, load_json_file, pandas_data_frame_columns_to_numpy
from kgcnn.mol.pymatgen import parse_cif_file_to_structures, convert_structures_as_dict, structure_get_properties, \
    structure_get_range_neighbors


class CrystalDataset(MemoryGraphDataset):
    """Class for making graph properties from periodic structures such as crystals.

    .. warning::
        Currently under construction.

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

    def __init__(self,
                 data_directory: str = None,
                 dataset_name: str = None,
                 file_name: str = None,
                 file_directory: str = None,
                 length: int = None,
                 verbose: int = 1, **kwargs):
        r"""Initialize a base class of :obj:`CrystalDataset`.

        Args:
            data_directory (str): Full path to directory of the dataset. Default is None.
            file_name (str): Filename for dataset to read into memory. This can be a single a 'cif' file.
                Or a '.csv' of file names that are expected to be cif-files in file_directory.
                Default is None.
            file_directory (str): Name or relative path from :obj:`data_directory` to a directory containing sorted
                'cif' files. Default is None.
            dataset_name (str): Name of the dataset. Important for naming and saving files. Default is None.
            length (int): Length of the dataset, if known beforehand. Default is None.
            verbose (int): Print progress or info for processing, where 0 is silent. Default is 1.
        """
        super(CrystalDataset, self).__init__(data_directory=data_directory, dataset_name=dataset_name,
                                             file_name=file_name, verbose=verbose, length=length,
                                             file_directory=file_directory, **kwargs)
        self.graph_lattice_matrix = None
        self.graph_abc = None
        self.graph_charge = None
        self.graph_volume = None
        self.range_image = None
        self._structs = None

    def _get_pymatgen_file_name(self):
        """Try to determine a file name for the pymatgen serialization information to store to disk."""
        return os.path.splitext(self.file_name)[0] + ".pymatgen.json"

    def prepare_data(self, cif_column_name: str = None, overwrite: bool = False):
        if os.path.exists(os.path.join(self.data_directory, self._get_pymatgen_file_name())) and not overwrite:
            print("INFO:kgcnn: Pickled pymatgen structures already exist. Do nothing.")
            return self
        pymatgen_file_made = False

        file_path = os.path.join(self.data_directory, self.file_name)
        file_path_base = os.path.splitext(file_path)[0]

        found_cif_file = False
        if os.path.exists(file_path_base + ".cif"):
            found_cif_file = True
            self.log("INFO:kgcnn: Start to read many structures form cif-file via pymatgen ...", end='', flush=True)
            structs = parse_cif_file_to_structures(file_path)
            self.log("done")
            self.log("INFO:kgcnn: Exporting as dict for pymatgen ...", end='', flush=True)
            dicts = convert_structures_as_dict(structs)
            self.log("done")
            self.log("INFO:kgcnn: Saving structures as .json ...", end='', flush=True)
            out_path = os.path.join(self.data_directory, self._get_pymatgen_file_name())
            save_json_file(dicts, out_path)
            self.log("done")
            pymatgen_file_made = True

        # We try to read in a csv file.
        self.read_in_table_file(file_path=file_path)

        if not found_cif_file and cif_column_name is not None and self.data_frame is not None:
            # Try to find file names in data_frame
            cif_file_list = self.data_frame[cif_column_name].values
            num_structs = len(cif_file_list)
            structs = []
            self.log("INFO:kgcnn: Read %s cif-file via pymatgen ..." % num_structs)
            for i, x in enumerate(cif_file_list):
                # Only one file per path
                structs.append(parse_cif_file_to_structures(os.path.join(self.data_directory,
                                                                         self.file_directory, x))[0])
                if i % 1000 == 0:
                    self.log(" ... read structure {0} from {1}".format(i, num_structs))
            self.log("done")
            self.log("INFO:kgcnn: Exporting as dict for pymatgen ...", end='', flush=True)
            dicts = convert_structures_as_dict(structs)
            self.log("done")
            self.log("INFO:kgcnn: Saving structures as .json ...", end='', flush=True)
            out_path = os.path.join(self.data_directory, self._get_pymatgen_file_name())
            save_json_file(dicts, out_path)
            self.log("done")
            pymatgen_file_made = True

        if not pymatgen_file_made:
            raise FileNotFoundError("ERROR:kgcnn: Could not make pymatgen structures.")

        return self

    def _read_pymatgen_json_in_memory(self):
        file_path = os.path.join(self.data_directory, self._get_pymatgen_file_name())
        if not os.path.exists(file_path):
            raise FileNotFoundError("ERROR:kgcnn: Cannot find .json file for `CrystalDataset`. Please prepare_data().")
        self.log("INFO:kgcnn: Reading structures from .json ...", end='', flush=True)
        dicts = load_json_file(file_path)
        structs = [pymatgen.core.structure.Structure.from_dict(x) for x in dicts]
        self.log("done")
        return structs

    def read_in_memory(self, label_column_name: str = None):

        file_path = os.path.join(self.data_directory, self.file_name)
        self.read_in_table_file(file_path=file_path)

        # We can try to get labels here.
        if self.data_frame is not None and label_column_name is not None:
            self.graph_labels = pandas_data_frame_columns_to_numpy(self.data_frame, label_column_name, "ERROR:kgcnn: ")

        structs = self._read_pymatgen_json_in_memory()
        self._structs = structs
        self.log("INFO:kgcnn: Making node features ...", end='', flush=True)
        node_symbol = []
        node_coordinates = []
        graph_lattice_matrix = []
        graph_abc = []
        graph_charge = []
        graph_volume = []
        for x in structs:
            coordinates, lattice_matrix, abc, charge, volume, symbols = structure_get_properties(x)
            node_coordinates.append(coordinates)
            graph_lattice_matrix.append(lattice_matrix)
            graph_abc.append(abc)
            graph_charge.append(charge)
            graph_volume.append(volume)
            node_symbol.append(symbols)

        self.node_symbol = node_symbol
        self.node_coordinates = node_coordinates
        self.graph_lattice_matrix = graph_lattice_matrix
        self.graph_abc = graph_abc
        self.graph_charge = graph_charge
        self.graph_volume = graph_volume
        self.log("done")

        return self

    def _set_dataset_range_from_structures(self, structs, radius: float = 4.0, numerical_tol: float = 1e-08,
                                           max_neighbours: int = 100000000):
        self.log("INFO:kgcnn: Setting range ...", end='', flush=True)
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
        self.log("done")

    def set_range(self, max_distance: float = 4.0, max_neighbours=15, do_invert_distance=False,
                  self_loops=True, exclusive=True):
        assert exclusive, "ERROR:kgcnn: Range in `CrystalDataset` only for exclusive=True at the moment."
        structs = self._read_pymatgen_json_in_memory()
        self._set_dataset_range_from_structures(structs=structs, radius=max_distance, max_neighbours=max_neighbours)
        return self

    def set_angle(self, prefix_indices: str = "range", compute_angles: bool = False):
        # Since coordinates are periodic.
        assert not compute_angles, "ERROR:kgcnn: Can not compute angles atm."
        super(CrystalDataset, self).set_angle(prefix_indices=prefix_indices, compute_angles=compute_angles)

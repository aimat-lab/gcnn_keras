import os
import pymatgen.core.structure

from kgcnn.data.base import MemoryGeometricGraphDataset
from kgcnn.utils.data import save_json_file, load_json_file
from kgcnn.mol.periodic import parse_cif_file_to_structures, convert_structures_as_dict, structure_get_properties, \
    structure_get_range_neighbors


class CrystalDataset(MemoryGeometricGraphDataset):
    global_proton_dict = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11,
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
    inverse_global_proton_dict = {value: key for key, value in global_proton_dict.items()}

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
            file_name (str): Generic filename for dataset to read into memory like a 'cif' file. Default is None.
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

    def _get_pymatgen_file_name(self):
        """Try to determine a file name for the pymatgen serialization information to store to disk."""
        return self.file_name[:self.file_name.rfind(
            ".")] + ".pymatgen.json" if "." in self.file_name else self.file_name + ".pymatgen.json"

    def prepare_data(self):
        file_path = os.path.join(self.data_directory, self.file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError("ERROR:kgcnn: Can not find file for `CrystalDataset`.")

        self._log("INFO:kgcnn: Start to read structures form cif-file via pymatgen ...", end='', flush=True)
        structs = parse_cif_file_to_structures(file_path)
        self._log("done")
        self._log("INFO:kgcnn: Exporting as dict for pymatgen ...", end='', flush=True)
        dicts = convert_structures_as_dict(structs)
        self._log("done")
        self._log("INFO:kgcnn: Saving structures as .json ...", end='', flush=True)
        out_path = os.path.join(self.data_directory, self._get_pymatgen_file_name())
        save_json_file(dicts, out_path)
        self._log("done")

    def _read_pymatgen_json_in_memory(self):
        file_path = os.path.join(self.data_directory, self._get_pymatgen_file_name())
        if not os.path.exists(file_path):
            raise FileNotFoundError("ERROR:kgcnn: Can not find file for `CrystalDataset`. Please run prepare_data().")
        self._log("INFO:kgcnn: Reading structures from .json ...", end='', flush=True)
        dicts = load_json_file(file_path)
        structs = [pymatgen.core.structure.Structure.from_dict(x) for x in dicts]
        self._log("done")
        return structs

    def read_in_memory(self):
        structs = self._read_pymatgen_json_in_memory()

        self._log("INFO:kgcnn: Making node features ...", end='', flush=True)
        node_number = []
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
            node_number.append([self.global_proton_dict[x] for x in symbols])

        self.node_number = node_number
        self.node_symbol = node_symbol
        self.node_coordinates = node_coordinates
        self.graph_lattice_matrix = graph_lattice_matrix
        self.graph_abc = graph_abc
        self.graph_charge = graph_charge
        self.graph_volume = graph_volume
        self._log("done")

        # We also compute for default range here.
        self._set_dataset_range_from_structures(structs)
        return self

    def _set_dataset_range_from_structures(self, structs, radius=4, numerical_tol: float = 1e-08):
        self._log("INFO:kgcnn: Setting range ...", end='', flush=True)
        range_indices = []
        range_image = []
        range_distance = []
        for x in structs:
            ridx, rimg, rd = structure_get_range_neighbors(x, radius=radius, numerical_tol=numerical_tol)
            range_indices.append(ridx)
            range_image.append(rimg)
            range_distance.append(rd)
        self.range_indices = range_indices
        self.range_image = range_image
        self.range_attributes = range_distance
        self._log("done")

    def set_range(self, max_distance=4, max_neighbours=15, do_invert_distance=False, self_loops=True, exclusive=True):
        print("WARNING:kgcnn: Range in `CrystalDataset` does not work for neighbours or self_loops yet.")
        structs = self._read_pymatgen_json_in_memory()
        self._set_dataset_range_from_structures(structs=structs, radius=max_distance)
        return self

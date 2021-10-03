import os
import pickle
import numpy as np
import json
# import shutil

from sklearn.preprocessing import StandardScaler
from kgcnn.data.qm import QMDataset
from kgcnn.mol.methods import ExtensiveMolecularScaler
from kgcnn.mol.convert import parse_mol_str
from kgcnn.utils.adj import add_edges_reverse_indices

class QM9Dataset(QMDataset):
    """Store and process QM9 dataset."""
    # https://ndownloader.figshare.com/files/3195398
    # https://ndownloader.figshare.com/files/3195389

    dataset_name = "QM9"
    data_main_dir = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets")
    data_directory = "qm9"
    download_url = "https://ndownloader.figshare.com/files/3195389"
    file_name = 'dsgdb9nsd.xyz.tar.bz2'
    unpack_tar = True
    unpack_zip = False
    unpack_directory = 'dsgdb9nsd.xyz'
    fits_in_memory = True
    require_prepare_data = True

    def __init__(self, reload: bool = False, verbose: int = 1):
        """Initialize QM9 dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Run base class default init()
        self.target_names = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H',
                             'G', 'Cv']
        super(QM9Dataset, self).__init__(reload=reload, verbose=verbose)
        if self.fits_in_memory:
            self.read_in_memory(verbose=verbose)

    def prepare_data(self, overwrite: bool = False, verbose: int = 1, **kwargs):
        """Process data by loading all single xyz-files and store all pickled information to file.
        The single files are deleted afterwards, requires to re-extract the tar-file for overwrite.

        Args:
            overwrite (bool): Whether to redo the processing, requires un-zip of the data again. Defaults to False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        path = os.path.join(self.data_main_dir, self.data_directory)

        datasetsize = 133885

        exist_pickle_dataset = False
        if (os.path.exists(os.path.join(path, "qm9.pickle")) or os.path.exists(
                os.path.join(path, "qm9.json"))) and not overwrite:
            if verbose > 0:
                print("INFO:kgcnn: Single molecules already pickled... done")
            exist_pickle_dataset = True

        qm9 = []
        if not exist_pickle_dataset:
            if not os.path.exists(os.path.join(path, 'dsgdb9nsd.xyz')):
                if verbose > 0:
                    print("ERROR:kgcnn: Can not find extracted dsgdb9nsd.xyz directory. Run extract dataset again.")
                return

            # Read individual files
            if verbose > 0:
                print("INFO:kgcnn: Reading dsgdb9nsd files ...", end='', flush=True)
            for i in range(1, datasetsize + 1):
                mol = []
                file = "dsgdb9nsd_" + "{:06d}".format(i) + ".xyz"
                open_file = open(os.path.join(path, "dsgdb9nsd.xyz", file), "r")
                lines = open_file.readlines()
                mol.append(int(lines[0]))
                labels = lines[1].strip().split(' ')[1].split('\t')
                if int(labels[0]) != i:
                    print("KGCNN:WARNING: Index for QM9 not matching xyz-file.")
                labels = [lines[1].strip().split(' ')[0].strip()] + [int(labels[0])] + [float(x) for x in labels[1:]]
                mol.append(labels)
                cords = []
                for j in range(int(lines[0])):
                    atom_info = lines[2 + j].strip().split('\t')
                    cords.append([atom_info[0]] + [float(x.replace('*^', 'e')) for x in atom_info[1:]])
                mol.append(cords)
                freqs = lines[int(lines[0]) + 2].strip().split('\t')
                freqs = [float(x) for x in freqs]
                mol.append(freqs)
                smiles = lines[int(lines[0]) + 3].strip().split('\t')
                mol.append(smiles)
                inchis = lines[int(lines[0]) + 4].strip().split('\t')
                mol.append(inchis)
                open_file.close()
                qm9.append(mol)
            if verbose > 0:
                print('done')

            # Save pickle data
            if verbose > 0:
                print("INFO:kgcnn: Saving qm9.json ...", end='', flush=True)
            with open(os.path.join(path, "qm9.json"), 'w') as f:
                json.dump(qm9, f)
            if verbose > 0:
                print('done')

            # Remove file after reading
            if verbose > 0:
                print("INFO:kgcnn: Cleaning up extracted files...", end='', flush=True)
            for i in range(1, datasetsize + 1):
                file = "dsgdb9nsd_" + "{:06d}".format(i) + ".xyz"
                file = os.path.join(path, "dsgdb9nsd.xyz", file)
                os.remove(file)
            if verbose > 0:
                print('done')

    def prepare_data_mol(self, overwrite: bool = False, verbose: int = 1, **kwargs):
        """Process data by making mol-objects. Does not work for all molecules. Also for around 10k molecules the
        string in the database does not match the mol-object generated from coordinates.

        Args:
            overwrite (bool): Whether to redo the processing, requires un-zip of the data again. Defaults to False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Check if we can import openbabel
        path = os.path.join(self.data_main_dir, self.data_directory)
        try:
            from openbabel import openbabel
            has_open_babel = True
        except ImportError:
            print("WARNING:kgcnn: Can not make mol-objects. Please install openbabel...", end='', flush=True)
            has_open_babel = False

        exist_mol_file = False
        if os.path.exists(os.path.join(path, self.mol_filename)) and not overwrite:
            if verbose > 0:
                print("INFO:kgcnn: Mol-object for molecules already created... done")
            exist_mol_file = True

        if not exist_mol_file and has_open_babel:
            if verbose > 0:
                print("INFO:kgcnn: Reading dataset...", end='', flush=True)
            if os.path.exists(os.path.join(path, "qm9.pickle")):
                with open(os.path.join(path, "qm9.pickle"), 'rb') as f:
                    qm9 = pickle.load(f)
            elif os.path.exists(os.path.join(path, "qm9.json")):
                with open(os.path.join(path, "qm9.json"), 'rb') as f:
                    qm9 = json.load(f)
            else:
                raise FileNotFoundError("Can not find pickled QM9 dataset.")
            if verbose > 0:
                print('done')

            # Try extract bond-info and save mol-file.
            if verbose > 0:
                print("INFO:kgcnn: Preparing bond information...", end='', flush=True)

            atoms = [x[2] for x in qm9]
            mol_list = self._make_mol_list(atoms)

            with open(os.path.join(path, self.mol_filename), 'w') as f:
                json.dump(mol_list, f)

            if verbose > 0:
                print('done')

    def read_in_memory(self, verbose=1):
        """Load the pickled QM9 data into memory and already split into items.

        Args:
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        path = os.path.join(self.data_main_dir, self.data_directory)

        if verbose > 0:
            print("INFO:kgcnn: Reading dataset ...", end='', flush=True)
        if os.path.exists(os.path.join(path, "qm9.pickle")):
            with open(os.path.join(path, "qm9.pickle"), 'rb') as f:
                qm9 = pickle.load(f)
        elif os.path.exists(os.path.join(path, "qm9.json")):
            with open(os.path.join(path, "qm9.json"), 'rb') as f:
                qm9 = json.load(f)
        else:
            raise FileNotFoundError("Can not find pickled QM9 dataset.")

        # labels
        self.length = 133885
        labels = np.array([x[1][1:] if len(x[1]) == 17 else x[1] for x in qm9])  # Remove 'gdb' tag here

        # Atoms as nodes
        atoms = [[y[0] for y in x[2]] for x in qm9]
        atom_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        zval = [[atom_dict[y] for y in x] for x in atoms]
        outzval = [np.array(x, dtype=np.int) for x in zval]
        nodes = outzval

        # Mean molecular weight mmw
        massdict = {'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984}
        mass = [[massdict[y] for y in x] for x in atoms]
        mmw = np.array([[np.mean(x), len(x)] for x in mass])

        # Coordinates
        coord = [[[y[1], y[2], y[3]] for y in x[2]] for x in qm9]
        coord = [np.array(x) for x in coord]

        self.graph_number = labels[:, 0]
        self.node_coordinates = coord
        self.graph_labels = labels[:, 1:]
        self.node_symbol = atoms
        self.node_number = nodes
        self.graph_attributes = mmw

        if verbose > 0:
            print('done')

        mol_list = None
        if os.path.exists(os.path.join(path, self.mol_filename)):
            if verbose > 0:
                print("INFO:kgcnn: Reading mol information ...", end='', flush=True)
            with open(os.path.join(path, self.mol_filename), 'rb') as f:
                mol_list =  json.load(f)
            print("done")
        else:
            print("WARNING:kgcnn: No mol information... done", flush=True)
        self.mol_list = mol_list
        if mol_list is not None:
            if verbose > 0:
                print("INFO:kgcnn: Parsing mol information ...", end='', flush=True)
            bond_info = []
            for x in mol_list:
                bond_info.append(np.array(parse_mol_str(x)[5], dtype="int"))
            edge_index = []
            edge_attr = []
            for x in bond_info:
                temp = add_edges_reverse_indices(np.array(x[:, :2]), np.array(x[:, 2:]))
                edge_index.append(temp[0])
                edge_attr.append(np.array(temp[1], dtype="float"))
            self.edge_indices = edge_index
            self.edge_attributes = edge_attr
            print("done")


class QM9GraphLabelScaler:
    """A standard scaler that scales all QM9 targets. For now, the main difference is that intensive and extensive
    properties are scaled differently. In principle, also dipole, polarizability or rotational constants
    could to be standardized differently."""

    def __init__(self, intensice_scaler=None, extensive_scaler=None):
        if intensice_scaler is None:
            intensice_scaler = {}
        if extensive_scaler is None:
            extensive_scaler = {}

        self.intensive_scaler = StandardScaler(**intensice_scaler)
        self.extensive_scaler = ExtensiveMolecularScaler(**extensive_scaler)

        self.scale_ = None

    def fit_transform(self, node_number, graph_labels):
        r"""Fit and transform all target labels for QM9.

        Args:
            node_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.
            graph_labels (np.ndarray): Array of QM9 labels of shape `(N, 15)`.

        Returns:
            np.ndarray: Transformed labels of shape `(N, 15)`.
        """
        self.fit(node_number, graph_labels)
        return self.transform(node_number, graph_labels)

    def transform(self, node_number, graph_labels):
        r"""Transform all target labels for QM9. Requires :obj:`fit()` called previously.

        Args:
            node_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.
            graph_labels (np.ndarray): Array of QM9 unscaled labels of shape `(N, 15)`.

        Returns:
            np.ndarray: Transformed labels of shape `(N, 15)`.
        """
        self._check_input(node_number, graph_labels)

        intensive_labels = graph_labels[:, :9]
        extensive_labels = graph_labels[:, 9:]

        trafo_intensive = self.intensive_scaler.transform(intensive_labels)
        trafo_extensive = self.extensive_scaler.transform(node_number, extensive_labels)

        out_labels = np.concatenate([trafo_intensive, trafo_extensive], axis=-1)
        return out_labels

    def fit(self, node_number, graph_labels):
        r"""Fit scaling of QM9 graph labels or targets.

        Args:
            node_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.
            graph_labels (np.ndarray): Array of QM9 labels of shape `(N, 15)`.

        Returns:
            self
        """
        self._check_input(node_number, graph_labels)

        # Note: Rotational Constants and r2 as well as dipole moment and polarizability
        # should be treated separately.
        intensive_labels = graph_labels[:, :9]
        extensive_labels = graph_labels[:, 9:]

        self.intensive_scaler.fit(intensive_labels)
        self.extensive_scaler.fit(node_number, extensive_labels)
        # print(self.intensive_scaler.scale_, self.extensive_scaler.scale_)
        self.scale_ = np.concatenate([self.intensive_scaler.scale_, self.extensive_scaler.scale_[0]], axis=0)
        return self

    def inverse_transform(self, node_number, graph_labels):
        r"""Back-transform all target labels for QM9.

        Args:
            node_number (list): List of atomic numbers for each molecule. E.g. `[np.array([6,1,1,1]), ...]`.
            graph_labels (np.ndarray): Array of QM9 scaled labels of shape `(N, 15)`.

        Returns:
            np.ndarray: Back-transformed labels of shape `(N, 15)`.
        """
        self._check_input(node_number, graph_labels)

        intensive_labels = graph_labels[:, :9]
        extensive_labels = graph_labels[:, 9:]

        inverse_trafo_intensive = self.intensive_scaler.inverse_transform(intensive_labels)
        inverse_trafo_extensive = self.extensive_scaler.inverse_transform(node_number, extensive_labels)

        out_labels = np.concatenate([inverse_trafo_intensive, inverse_trafo_extensive], axis=-1)
        return out_labels

    def padd(self, selected_targets, target_indices):
        r"""Padding a set of specific targets defined by `target_indices` to the full QM9 target dimension of 15.

        Args:
            selected_targets (np.ndarray): A reduced selection of QM9 target `(n_samples, n_targets)` where
                `n_targets` <= 15.
            target_indices (np.ndarray): Indices of specific targets of shape `(n_targets, )`.

        Returns:
            np.ndarray: Array of QM9 labels of shape `(N, 15)`.
        """
        labels = np.zeros((len(selected_targets), 15))
        labels[:, target_indices] = selected_targets
        return labels

    @staticmethod
    def _check_input(node_number, graph_labels):
        assert len(node_number) == len(graph_labels), "ERROR:kgcnn: `QM9GraphLabelScaler` needs same length input."
        assert graph_labels.shape[-1] == 15, "ERROR:kgcnn: `QM9GraphLabelScaler` got wrong targets."

# dataset = QM9Dataset()
# scaler = QM9GraphLabelScaler()
# tafo_labels = scaler.fit_transform(dataset.node_number, dataset.graph_labels)
# rev_labels = scaler.inverse_transform(dataset.node_number, tafo_labels
# print(np.amax(np.abs(dataset.graph_labels-rev_labels)))

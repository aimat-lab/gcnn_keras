import os
import pickle
import numpy as np
import json
# import shutil

from kgcnn.data.qm import QMDataset


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

    def __init__(self, reload=False, verbose=1):
        """Initialize QM9 dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Run base class default init()
        self.target_names = ['index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H',
                             'G', 'Cv']
        super(QM9Dataset, self).__init__(reload=reload, verbose=verbose)
        if self.fits_in_memory:
            self.read_in_memory(verbose=verbose)

    def prepare_data(self, overwrite=False, verbose=1, **kwargs):
        """Process data by loading all single xyz-files and store all pickled information to file.
        The single files are deleted afterwards, requires to re-extract the tar-file for overwrite.

        Args:
            overwrite (bool): Whether to redo the processing, requires un-zip of the data again. Defaults to False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.

        Returns:
            pickle: Pickled QM9 data as python list.
        """
        path = os.path.join(self.data_main_dir, self.data_directory)

        datasetsize = 133885
        qm9 = []

        if (os.path.exists(os.path.join(path, "qm9.pickle")) or os.path.exists(
                os.path.join(path, "qm9.json"))) and not overwrite:
            if verbose > 0:
                print("INFO:kgcnn: Single molecules already pickled... done")
            return qm9

        if not os.path.exists(os.path.join(path, 'dsgdb9nsd.xyz')):
            if verbose > 0:
                print("ERROR:kgcnn: Can not find extracted dsgdb9nsd.xyz directory. Run extract dataset again.")
            return qm9

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
        # if verbose > 0:
        #     print("INFO: Saving qm9.pickle ...", end='', flush=True)
        # with open(os.path.join(path, "qm9.pickle"), 'wb') as f:
        #     pickle.dump(qm9, f)
        # if verbose > 0:
        #     print('done')

        # Remove file after reading
        if verbose > 0:
            print("INFO:kgcnn: Cleaning up extracted files...", end='', flush=True)
        for i in range(1, datasetsize + 1):
            file = "dsgdb9nsd_" + "{:06d}".format(i) + ".xyz"
            file = os.path.join(path, "dsgdb9nsd.xyz", file)
            os.remove(file)
        if verbose > 0:
            print('done')

        return qm9

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
        # print(labels[0])
        # Atoms as nodes
        atoms = [[y[0] for y in x[2]] for x in qm9]
        # nodelens = np.array([len(x) for x in atoms], dtype=np.int)
        atom_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        # atom_1hot = {'H': [1, 0, 0, 0, 0], 'C': [0, 1, 0, 0, 0], 'N': [0, 0, 1, 0, 0], 'O': [0, 0, 0, 1, 0],
        #              'F': [0, 0, 0, 0, 1]}
        zval = [[atom_dict[y] for y in x] for x in atoms]
        outzval = [np.array(x, dtype=np.int) for x in zval]
        # outatoms = np.concatenate(outatom,axis=0)
        # a1hot = [[atom_1hot[y] for y in x] for x in atoms]
        # outa1hot = [np.array(x, dtype=np.float32) for x in a1hot]
        nodes = outzval

        # Mean molecular weight mmw
        massdict = {'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984}
        mass = [[massdict[y] for y in x] for x in atoms]
        mmw = np.array([[np.mean(x), len(x)] for x in mass])

        # Coordinates
        coord = [[[y[1], y[2], y[3]] for y in x[2]] for x in qm9]
        coord = [np.array(x) for x in coord]

        self.node_coordinates = coord
        self.graph_labels = labels
        self.node_symbol = atoms
        self.node_number = nodes
        self.graph_attributes = mmw

        if verbose > 0:
            print('done')


class QM9Scaler:

    def __init__(self, node_coordinates, node_number, graph_labels):
        self.node_coordinates = node_coordinates
        self.graph_labels = graph_labels
        self.node_number = node_number

        self.intensive_scaler = None
        self.extensive_scaler = None

    def fit_transform(self, training_indices):
        self.fit(training_indices)

    def transform(self, labels):
        pass

    def fit(self, training_indices):
        """Fit scaling of labels using indices."""
        # Pick Training selection.
        label_selection = self.graph_labels[training_indices]

        # Note: Rotational Constants and r2 as well as dipole moment and polarizability
        # should be treated separately.
        intensive_labels = label_selection[:, :8]
        extensive_labels = label_selection[:, 8:]


    def inverse_transform(self, labels):
        pass


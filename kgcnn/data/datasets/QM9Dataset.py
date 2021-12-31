import os
import pickle
import numpy as np
import json

from kgcnn.data.qm import QMDataset
from kgcnn.data.download import DownloadDataset
from kgcnn.mol.io import write_list_to_xyz_file


class QM9Dataset(QMDataset, DownloadDataset):
    """Store and process QM9 dataset."""

    download_info = {
        "dataset_name": "QM9",
        "data_directory_name": "qm9",
        # https://ndownloader.figshare.com/files/3195398
        "download_url": "https://ndownloader.figshare.com/files/3195389",
        "download_file_name": 'dsgdb9nsd.xyz.tar.bz2',
        "unpack_tar": True,
        "unpack_zip": False,
        "unpack_directory_name": 'dsgdb9nsd.xyz',
    }

    def __init__(self, reload: bool = False, verbose: int = 1):
        """Initialize QM9 dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        QMDataset.__init__(self, verbose=verbose, dataset_name="QM9")
        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)

        self.label_names = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H',
                             'G', 'Cv']
        self.label_units = ["GHz", "GHz", "GHz", "D", r"a_0^3", "eV", "eV", "eV", r"a_0^2", "eV", "eV", "eV", "eV",
                             "eV", r"cal/mol K"]
        self.label_unit_conversion = np.array(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 27.2114, 27.2114, 27.2114, 1.0, 27.2114, 27.2114, 27.2114,
              27.2114, 27.2114, 1.0]]
        )  # Pick always same units for training
        self.dataset_name = "QM9"
        self.require_prepare_data = True
        self.fits_in_memory = True
        self.verbose = verbose
        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name)
        self.file_name = "qm9.xyz"

        if self.require_prepare_data:
            self.prepare_data(overwrite=reload)

        if self.fits_in_memory:
            self.read_in_memory()

    def prepare_data(self, overwrite: bool = False, xyz_column_name: str = None, make_sdf: bool = True):
        """Process data by loading all single xyz-files and store all pickled information to file.
        The single files are deleted afterwards, requires to re-extract the tar-file for overwrite.

        Args:
            overwrite (bool): Whether to redo the processing, requires un-zip of the data again. Defaults to False.
            xyz_column_name (str): Not used.
            make_sdf (bool): Whether to make SDF file.
        """
        path = self.data_directory

        dataset_size = 133885

        if (os.path.exists(os.path.join(path, "qm9.pickle")) or os.path.exists(
                os.path.join(path, "qm9.json"))) and not overwrite:
            self.info("Single molecules already pickled.")
        else:
            if not os.path.exists(os.path.join(path, 'dsgdb9nsd.xyz')):
                self.error("Can not find extracted dsgdb9nsd.xyz directory. Run reload dataset again.")
                return
            qm9 = []
            # Read individual files
            self.info("Reading dsgdb9nsd files ...")
            for i in range(1, dataset_size + 1):
                mol = []
                file = "dsgdb9nsd_" + "{:06d}".format(i) + ".xyz"
                open_file = open(os.path.join(path, "dsgdb9nsd.xyz", file), "r")
                lines = open_file.readlines()
                mol.append(int(lines[0]))
                labels = lines[1].strip().split(' ')[1].split('\t')
                if int(labels[0]) != i:
                    self.warning("Index for QM9 not matching xyz-file.")
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

            # Save pickle data
            self.info("Saving qm9.json ...")
            with open(os.path.join(path, "qm9.json"), 'w') as f:
                json.dump(qm9, f)

            # Remove file after reading
            self.info("Cleaning up extracted files...")
            for i in range(1, dataset_size + 1):
                file = "dsgdb9nsd_" + "{:06d}".format(i) + ".xyz"
                file = os.path.join(path, "dsgdb9nsd.xyz", file)
                os.remove(file)

        if os.path.exists(os.path.join(path, self.file_name)) and not overwrite:
            self.info("Single xyz-file %s for molecules already created." % self.file_name)
        else:
            self.info("Reading dataset...")
            if os.path.exists(os.path.join(path, "qm9.pickle")):
                with open(os.path.join(path, "qm9.pickle"), 'rb') as f:
                    qm9 = pickle.load(f)
            elif os.path.exists(os.path.join(path, "qm9.json")):
                with open(os.path.join(path, "qm9.json"), 'rb') as f:
                    qm9 = json.load(f)
            else:
                raise FileNotFoundError("Can not find pickled QM9 dataset.")

            # Try extract bond-info and save mol-file.
            self.info("Writing single xyz-file ...")
            pos = [[y[1:] for y in x[2]] for x in qm9]
            atoms = [[y[0] for y in x[2]] for x in qm9]
            atoms_pos = [[x, y] for x, y in zip(atoms, pos)]
            write_list_to_xyz_file(os.path.join(path, "qm9.xyz"), atoms_pos)

        super(QM9Dataset, self).prepare_data(overwrite=overwrite)
        return self

    def read_in_memory(self, label_column_name: str = None):
        """Load the pickled QM9 data into memory and already split into items.

        Args:
            label_column_name(str): Not used.

        Returns:
            self
        """
        path = self.data_directory

        self.info("Reading dataset ...")
        if os.path.exists(os.path.join(path, "qm9.pickle")):
            with open(os.path.join(path, "qm9.pickle"), 'rb') as f:
                qm9 = pickle.load(f)
        elif os.path.exists(os.path.join(path, "qm9.json")):
            with open(os.path.join(path, "qm9.json"), 'rb') as f:
                qm9 = json.load(f)
        else:
            raise FileNotFoundError("Can not find pickled QM9 dataset.")

        # labels
        # self.length = 133885
        labels = np.array([x[1][1:] if len(x[1]) == 17 else x[1] for x in qm9])  # Remove 'gdb' tag here

        # Atoms as nodes
        atoms = [[y[0] for y in x[2]] for x in qm9]
        atom_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        zval = [[atom_dict[y] for y in x] for x in atoms]
        outzval = [np.array(x, dtype="int") for x in zval]
        nodes = outzval

        # Mean molecular weight mmw
        massdict = {'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984}
        mass = [[massdict[y] for y in x] for x in atoms]
        mmw = np.array([[np.mean(x), len(x)] for x in mass])

        # Coordinates
        coord = [[[y[1], y[2], y[3]] for y in x[2]] for x in qm9]
        coord = [np.array(x) for x in coord]

        # Labels
        ids = [x for x in labels[:, 0]]
        actual_labels = self.label_unit_conversion*labels[:, 1:]
        actual_labels = [x for x in actual_labels]

        self.assign_property("graph_number", ids)
        self.assign_property("node_coordinates", coord)
        self.assign_property("graph_labels", actual_labels)
        self.assign_property("node_symbol", atoms)
        self.assign_property("node_number", nodes)
        self.assign_property("graph_attributes", [x for x in mmw])

        # Try to read mol information
        self.read_in_memory_sdf()
        return self


# from kgcnn.data.qm import QMGraphLabelScaler
# dataset = QM9Dataset()
# scaler = QMGraphLabelScaler([{"class_name": "StandardScaler", "config": {}},
#                              {"class_name": "ExtensiveMolecularScaler", "config": {}},
#                              {"class_name": "ExtensiveMolecularScaler", "config": {}}])
# trafo_labels = scaler.fit_transform(np.array(dataset.graph_labels)[:, :3], dataset.node_number)
# rev_labels = scaler.inverse_transform(dataset.node_number, tafo_labels
# print(np.amax(np.abs(dataset.graph_labels-rev_labels)))

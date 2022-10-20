import os
import pickle
import numpy as np
import json
import pandas as pd
from typing import Union
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
        self.file_name = "qm9.csv"

        # We try to download also additional files here.
        self.download_database(path=self.data_directory, filename="readme.txt", logger=None,
                               download_url="https://figshare.com/ndownloader/files/3195392", overwrite=reload)
        self.download_database(path=self.data_directory, filename="uncharacterized.txt", logger=None,
                               download_url="https://figshare.com/ndownloader/files/3195404", overwrite=reload)
        self.download_database(path=self.data_directory, filename="atomref.txt", logger=None,
                               download_url="https://figshare.com/ndownloader/files/3195395", overwrite=reload)
        self.download_database(path=self.data_directory, filename="validation.txt", logger=None,
                               download_url="https://figshare.com/ndownloader/files/3195401", overwrite=reload)

        if self.require_prepare_data:
            self.prepare_data(overwrite=reload)

        if self.fits_in_memory:
            self.read_in_memory(label_column_name=self.label_names)

    def prepare_data(self, overwrite: bool = False, file_column_name: str = None, make_sdf: bool = True):
        """Process data by loading all single xyz-files and store all pickled information to file.
        The single files are deleted afterwards, requires to re-extract the tar-file for overwrite.

        Args:
            overwrite (bool): Whether to redo the processing, requires un-zip of the data again. Defaults to False.
            file_column_name (str): Not used.
            make_sdf (bool): Whether to make SDF file.
        """
        path = self.data_directory
        dataset_size = 133885

        if os.path.exists(self.file_path) and os.path.exists(self.file_path_xyz) and not overwrite:
            self.info("Single XYZ file and CSV table with labels already created.")
            return super(QM9Dataset, self).prepare_data(overwrite=overwrite)

        # Reading files.
        if not os.path.exists(os.path.join(path, 'dsgdb9nsd.xyz')):
            self.error("Can not find extracted dsgdb9nsd.xyz directory. Run reload QM9 dataset again.")
            return self

        # Read individual files
        qm9 = []
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

        # Save pickled data.
        self.info("Saving qm9.json ...")
        with open(os.path.join(path, "qm9.json"), 'w') as f:
            json.dump(qm9, f)

        # Remove file after reading
        self.info("Cleaning up extracted files...")
        for i in range(1, dataset_size + 1):
            file = "dsgdb9nsd_" + "{:06d}".format(i) + ".xyz"
            file = os.path.join(path, "dsgdb9nsd.xyz", file)
            os.remove(file)

        # Creating XYZ file.
        self.info("Writing single xyz-file ...")
        pos = [[y[1:] for y in x[2]] for x in qm9]
        atoms = [[y[0] for y in x[2]] for x in qm9]
        atoms_pos = [[x, y] for x, y in zip(atoms, pos)]
        write_list_to_xyz_file(os.path.join(path, "qm9.xyz"), atoms_pos)

        # Creating Table file with labels.
        labels = np.array([x[1][1:] if len(x[1]) == 17 else x[1] for x in qm9])  # Remove 'gdb' tag here
        # ids = [x for x in labels[:, 0]]
        actual_labels = self.label_unit_conversion*labels[:, 1:]
        actual_labels = [x for x in actual_labels]
        targets = pd.DataFrame(actual_labels, columns=self.label_names)
        self.info("Writing CSV file of graph labels.")
        targets.to_csv(self.file_path, index=False)

        return super(QM9Dataset, self).prepare_data(
            overwrite=overwrite, file_column_name=file_column_name, make_sdf=make_sdf)

    def read_in_memory(self, **kwargs):
        super(QM9Dataset, self).read_in_memory(**kwargs)

        # Mean molecular weight mmw
        mass_dict = {'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984, 'S': 32.065, "C3": 12.0107}

        def mmw(atoms):
            mass = [mass_dict[x[:1]] for x in atoms]
            return np.array([np.mean(mass), len(mass)])

        self.assign_property("graph_attributes", [mmw(x) for x in self.obtain_property("node_symbol")])
        return self


# from kgcnn.data.qm import QMGraphLabelScaler
# dataset = QM9Dataset(reload=False)
# scaler = QMGraphLabelScaler([{"class_name": "StandardScaler", "config": {}},
#                              {"class_name": "ExtensiveMolecularScaler", "config": {}},
#                              {"class_name": "ExtensiveMolecularScaler", "config": {}}])
# trafo_labels = scaler.fit_transform(np.array(dataset.graph_labels)[:, :3], dataset.node_number)
# rev_labels = scaler.inverse_transform(dataset.node_number, tafo_labels
# print(np.amax(np.abs(dataset.graph_labels-rev_labels)))

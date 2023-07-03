import os
# import pickle
import numpy as np
import json
import pandas as pd
# from typing import Union
from kgcnn.data.qm import QMDataset
from kgcnn.data.download import DownloadDataset
from kgcnn.molecule.io import write_list_to_xyz_file


class QM9Dataset(QMDataset, DownloadDataset):
    r"""Store and process QM9 dataset from `Quantum Machine <http://quantum-machine.org/datasets/>`__ . dataset.

    Dataset of 134k stable small organic molecules made up of C,H,O,N,F.

    From `Quantum Machine <http://quantum-machine.org/datasets/>`__ :
    Computational de novo design of new drugs and materials requires rigorous and unbiased exploration of chemical
    compound space. However, large uncharted territories persist due to its size scaling combinatorially with
    molecular size. We report computed geometric, energetic, electronic, and thermodynamic properties for 134k stable
    small organic molecules made up of CHONF. These molecules correspond to the subset of all 133,885 species with up
    to nine heavy atoms (CONF) out of the GDB-17 chemical universe of 166 billion organic molecules. We report
    geometries minimal in energy, corresponding harmonic frequencies, dipole moments, polarizabilities, along
    with energies, enthalpies, and free energies of atomization. All properties were calculated at
    the B3LYP/6-31G(2df,p) level of quantum chemistry. Furthermore, for the predominant stoichiometry, C7H10O2,
    there are 6,095 constitutional isomers among the 134k molecules. We report energies, enthalpies, and free
    energies of atomization at the more accurate G4MP2 level of theory for all of them. As such, this data set provides
    quantum chemical properties for a relevant, consistent, and comprehensive chemical space of small organic molecules.
    This database may serve the benchmarking of existing methods, development of new methods, such as hybrid quantum
    mechanics/machine learning, and systematic identification of structure-property relationships.

    Labels include geometric, energetic, electronic, and thermodynamic properties. Typically, a random 10% validation
    and 10% test set are used. In literature, test errors are given as MAE and for energies are in [eV].

    Molecules that have a different smiles code after convergence can be removed with :obj:`remove_uncharacterized` .
    Also labels with removed atomization energy are generated.

    .. code-block:: python

        from kgcnn.data.datasets.QM9Dataset import QM9Dataset
        dataset = QM9Dataset(reload=True)
        print(dataset[0])

    References:

        (1) L. Ruddigkeit, R. van Deursen, L. C. Blum, J.-L. Reymond, Enumeration of 166 billion organic small
            molecules in the chemical universe database GDB-17, J. Chem. Inf. Model. 52, 2864–2875, 2012.
        (2) R. Ramakrishnan, P. O. Dral, M. Rupp, O. A. von Lilienfeld, Quantum chemistry structures and properties
            of 134 kilo molecules, Scientific Data 1, 140022, 2014.

    """

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

    def __init__(self, reload: bool = False, verbose: int = 10):
        """Initialize QM9 dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        QMDataset.__init__(self, verbose=verbose, dataset_name="QM9")
        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)

        self.label_names = [
            'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv',
            'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'Cv_atom']
        self.label_units = [
            "GHz", "GHz", "GHz", "D", r"a_0^3", "eV", "eV", "eV", r"a_0^2", "eV", "eV", "eV", "eV", "eV", r"cal/mol K",
            "eV", "eV", "eV", "eV", r"cal/mol K"]
        self.label_unit_conversion = np.array(
            [[1.0, 1.0, 1.0, 1.0, 1.0, 27.2114, 27.2114, 27.2114, 1.0, 27.2114, 27.2114, 27.2114, 27.2114, 27.2114, 1.0,
              27.2114, 27.2114, 27.2114, 27.2114, 1.0]]
        )  # Pick always same units for training
        self.dataset_name = "QM9"
        self.require_prepare_data = True
        self.fits_in_memory = True
        self.verbose = verbose
        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name)
        self.file_name = "qm9.csv"
        self.__removed_uncharacterized = False
        self.__atom_ref = {
            # units are in hartree of original dataset.
            "U0": {"H": -0.500273, "C": -37.846772, "N": -54.583861, "O": -75.064579, "F": -99.718730},
            "U": {"H": -0.498857, "C": -37.845355, "N": -54.582445, "O": -75.063163, "F": -99.717314},
            "H": {"H": -0.497912, "C": -37.844411, "N": -54.581501, "O": -75.062219, "F": -99.716370},
            "G": {"H": -0.510927, "C": -37.861317, "N": -54.598897, "O": -75.079532, "F": -99.733544},
            "CV": {"H": 2.981, "C": 2.981, "N": 2.981, "O": 2.981, "F": 2.981},
        }

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
            self.read_in_memory(
                label_column_name=["%s [%s]" % (a, b) for a, b in zip(self.label_names, self.label_units)])

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
            return super(QM9Dataset, self).prepare_data(
                overwrite=overwrite, file_column_name=file_column_name, make_sdf=make_sdf)

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
        pos = [[y[1:4] for y in x[2]] for x in qm9]
        atoms = [[y[0] for y in x[2]] for x in qm9]
        atoms_pos = [[x, y] for x, y in zip(atoms, pos)]
        write_list_to_xyz_file(os.path.join(path, "qm9.xyz"), atoms_pos)

        # Creating Table file with labels.
        labels = np.array([x[1][1:] if len(x[1]) == 17 else x[1] for x in qm9])  # Remove 'gdb' tag if not done already.
        atom_energy = [[sum([self.__atom_ref[t][a] for a in x]) for t in ["U0", "U", "H", "G", "CV"]] for x in atoms]
        targets_atom = labels[:, 11:] - np.array(atom_energy)
        targets = np.concatenate([labels[:, 1:], targets_atom], axis=-1)*self.label_unit_conversion
        df = pd.DataFrame(targets, columns=["%s [%s]" % (a, b) for a, b in zip(self.label_names, self.label_units)])
        df.insert(targets.shape[1], "ID", labels[:, 0].astype(dtype="int"))  # add id at the end for reference.
        self.info("Writing CSV file of graph labels.")
        df.to_csv(self.file_path, index=False)

        return super(QM9Dataset, self).prepare_data(
            overwrite=overwrite, file_column_name=file_column_name, make_sdf=make_sdf)

    def read_in_memory(self, **kwargs):
        super(QM9Dataset, self).read_in_memory(**kwargs)

        # Mean molecular weight mmw
        mass_dict = {'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984, 'S': 32.065, "C3": 12.0107}

        def mmw(atoms):
            mass = [mass_dict[x[:1]] for x in atoms]
            return np.array([np.mean(mass), len(mass)])

        # TODO: Do this in graph_attributes mol interface.
        self.assign_property("graph_attributes", [
            mmw(x) if x is not None else None for x in self.obtain_property("node_symbol")])
        return self

    def remove_uncharacterized(self):
        """Remove 3054 uncharacterized molecules that failed structure test from this dataset."""
        if self.__removed_uncharacterized:
            self.error("Uncharacterized molecules have already been removed. Continue.")
            return
        with open(os.path.join(self.data_directory, "uncharacterized.txt"), "r") as f:
            data = f.readlines()[9:-1]
        data = [x.strip().split(" ") for x in data]
        data = [[y for y in x if y != ""] for x in data]
        indices = np.array([x[0] for x in data], dtype="int") - 1
        indices_backward = np.flip(np.sort(indices))
        for i in indices_backward:
            self.pop(int(i))
        self.info("Removed %s uncharacterized molecules." % len(indices_backward))
        self.__removed_uncharacterized = True
        return indices_backward

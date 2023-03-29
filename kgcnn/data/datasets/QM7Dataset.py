import os
import pickle
import numpy as np
import scipy.io
import json
import pandas as pd
from typing import Union
from kgcnn.data.qm import QMDataset
from kgcnn.data.download import DownloadDataset
from kgcnn.molecule.io import write_list_to_xyz_file
from kgcnn.graph.methods import coulomb_matrix_to_inverse_distance_proton, coordinates_from_distance_matrix
from kgcnn.graph.methods import invert_distance
from kgcnn.molecule.methods import inverse_global_proton_dict


class QM7Dataset(QMDataset, DownloadDataset):
    r"""Store and process QM7 dataset from `Quantum Machine <http://quantum-machine.org/datasets/>`__ . dataset.

    From `Quantum Machine <http://quantum-machine.org/datasets/>`__ :
    This dataset is a subset of GDB-13 (a database of nearly 1 billion stable and synthetically accessible
    organic molecules) composed of all molecules of up to 23 atoms (including 7 heavy atoms C, N, O, and S),
    totalling 7165 molecules. We provide the Coulomb matrix representation of these molecules and their atomization
    energies computed similarly to the FHI-AIMS implementation of the Perdew-Burke-Ernzerhof hybrid functional (PBE0).
    This dataset features a large variety of molecular structures such as double and triple bonds, cycles, carboxy,
    cyanide, amide, alcohol and epoxy.
    The atomization energies are given in kcal/mol and are ranging from -800 to -2000 kcal/mol.
    The dataset is composed of three multidimensional arrays X (7165 x 23 x 23), Tm(7165) and P (5 x 1433)
    representing the inputs (Coulomb matrices), the labels (atomization energies) and the splits for cross-validation,
    respectively. The dataset also contain two additional multidimensional arrays Z (7165) and R (7165 x 3)
    representing the atomic charge and the cartesian coordinate of each atom in the molecules.

    Here, the coordinates are given and converted with :obj:`QMDataset` to molecular structure.
    Labels are not scaled but have original units. Original splits are added to the dataset.

    References:

        (1) L. C. Blum, J.-L. Reymond, 970 Million Druglike Small Molecules for Virtual Screening in the Chemical
            Universe Database GDB-13, J. Am. Chem. Soc., 131:8732, 2009.
        (2) M. Rupp, A. Tkatchenko, K.-R. Müller, O. A. von Lilienfeld: Fast and Accurate Modeling of Molecular
            Atomization Energies with Machine Learning, Physical Review Letters, 108(5):058301, 2012.

    """

    download_info = {
        "dataset_name": "QM7",
        "data_directory_name": "qm7",
        # https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.mat
        "download_url": "http://quantum-machine.org/data/qm7.mat",
        "download_file_name": 'qm7.mat',
        "unpack_tar": False,
        "unpack_zip": False,
    }

    def __init__(self, reload: bool = False, verbose: int = 10):
        """Initialize QM9 dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        QMDataset.__init__(self, verbose=verbose, dataset_name="QM7")
        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)

        self.label_names = ["u0_atom"]
        self.label_units = ["kcal/mol"]
        self.label_unit_conversion = np.array([1.0] * 14)  # Pick always same units for training
        self.dataset_name = "QM7"
        self.require_prepare_data = True
        self.fits_in_memory = True
        self.verbose = verbose
        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name)
        self.file_name = "qm7.csv"

        if self.require_prepare_data:
            self.prepare_data(overwrite=reload)

        if self.fits_in_memory:
            self.read_in_memory(label_column_name=self.label_names)

    def prepare_data(self, overwrite: bool = False, file_column_name: str = None, make_sdf: bool = True):

        if not os.path.exists(self.file_path_xyz) or overwrite:
            mat = scipy.io.loadmat(os.path.join(self.data_directory, self.download_info["download_file_name"]))
            graph_len = [int(np.around(np.sum(x > 0))) for x in mat["Z"]]
            proton = [x[:i] for i, x in zip(graph_len, mat["Z"])]
            atoms = [[inverse_global_proton_dict[i] for i in x] for x in proton]
            pos = [x[:i, :]*0.529177210903 for i, x in zip(graph_len, mat["R"])]
            atoms_pos = [[x, y] for x, y in zip(atoms, pos)]
            np.save(os.path.join(self.data_directory, "qm7_splits.npy"), mat["P"])
            self.info("Writing XYZ file from coulomb matrix information.")
            write_list_to_xyz_file(self.file_path_xyz, atoms_pos)
        else:
            self.info("Found XYZ file for qm7b already created.")

        if not os.path.exists(self.file_path) or overwrite:
            mat = scipy.io.loadmat(os.path.join(self.data_directory, self.download_info["download_file_name"]))
            labels = mat["T"][0]
            targets = pd.DataFrame(labels, columns=self.label_names)
            self.info("Writing CSV file of graph labels.")
            targets.to_csv(self.file_path, index=False)
        else:
            self.info("Found CSV file of graph labels.")

        return super(QM7Dataset, self).prepare_data(
            overwrite=overwrite, file_column_name=file_column_name, make_sdf=make_sdf)

    def _get_cross_validation_splits(self):
        return np.load(os.path.join(self.data_directory, "qm7_splits.npy"))

    def read_in_memory(self, **kwargs):
        super(QM7Dataset, self).read_in_memory( **kwargs)
        splits = self._get_cross_validation_splits()
        property_split = []
        for i in range(len(self)):
            is_in_split = []
            for j, split in enumerate(splits):
                if i in split:
                    is_in_split.append(j)
            property_split.append(np.array(is_in_split, dtype="int"))
        self.assign_property("kfold", property_split)

        # Mean molecular weight mmw
        mass_dict = {'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984, 'S': 32.065, "C3": 12.0107}

        def mmw(atoms):
            mass = [mass_dict[x[:1]] for x in atoms]
            return np.array([np.mean(mass), len(mass)])

        # TODO: Do this in graph_attributes mol interface.
        self.assign_property("graph_attributes",
                             [mmw(x) if x is not None else None for x in self.obtain_property("node_symbol")])

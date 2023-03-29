import os
import numpy as np
import scipy.io
import pandas as pd

from kgcnn.data.qm import QMDataset
from kgcnn.data.download import DownloadDataset
from kgcnn.molecule.io import write_list_to_xyz_file
from kgcnn.graph.methods import coulomb_matrix_to_inverse_distance_proton, coordinates_from_distance_matrix
from kgcnn.graph.methods import invert_distance
from kgcnn.molecule.methods import inverse_global_proton_dict


class QM7bDataset(QMDataset, DownloadDataset):
    r"""Store and process QM7b dataset from `Quantum Machine <http://quantum-machine.org/datasets/>`__ .

    From `Quantum Machine <http://quantum-machine.org/datasets/>`__ :
    This dataset is an extension of the QM7 dataset for multitask learning where 13 additional properties
    (e.g. polarizability, HOMO and LUMO eigenvalues, excitation energies) have to be predicted at different
    levels of theory (ZINDO, SCS, PBE0, GW). Additional molecules comprising chlorine atoms are also included,
    totalling 7211 molecules.

    The dataset is composed of two multidimensional arrays X (7211 x 23 x 23) and T (7211 x 14) representing the inputs
    (Coulomb matrices) and the labels (molecular properties) and one array names of size 14 listing the names of the
    different properties.

    Here, the Coulomb matrices are converted back into coordinates and with :obj:`QMDataset` to molecular structure.
    Labels are not scaled but have original units.

    References:

        (1) L. C. Blum, J.-L. Reymond, 970 Million Druglike Small Molecules for Virtual Screening in
            the Chemical Universe Database GDB-13, J. Am. Chem. Soc., 131:8732, 2009.
        (2) G. Montavon, M. Rupp, V. Gobre, A. Vazquez-Mayagoitia, K. Hansen, A. Tkatchenko, K.-R. Müller,
            O.A. von Lilienfeld, Machine Learning of Molecular Electronic Properties in Chemical Compound Space,
            New J. Phys. 15 095003, 2013.

    """

    download_info = {
        "dataset_name": "QM7b",
        "data_directory_name": "qm7b",
        # https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7b.mat
        "download_url": "http://quantum-machine.org/data/qm7b.mat",
        "download_file_name": 'qm7b.mat',
        "unpack_tar": False,
        "unpack_zip": False,
    }

    def __init__(self, reload: bool = False, verbose: int = 10):
        """Initialize QM9 dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        QMDataset.__init__(self, verbose=verbose, dataset_name="QM7b")
        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)

        self.label_names = ["aepbe0", "zindo-excitation-energy-with-the-most-absorption",
                            "zindo-highest-absorption", "zindo-homo", "zindo-lumo",
                            "zindo-1st-excitation-energy", "zindo-ionization-potential", "zindo-electron-affinity",
                            "ks-homo", "ks-lumo", "gw-homo", "gw-lumo", "polarizability-pbe", "polarizability-scs"]
        self.label_units = ["[?]"] * 14
        self.label_unit_conversion = np.array([1.0] * 14)  # Pick always same units for training
        self.dataset_name = "QM7b"
        self.require_prepare_data = True
        self.fits_in_memory = True
        self.verbose = verbose
        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name)
        self.file_name = "qm7b.csv"

        if self.require_prepare_data:
            self.prepare_data(overwrite=reload)

        if self.fits_in_memory:
            self.read_in_memory(label_column_name=self.label_names)

    def prepare_data(self, overwrite: bool = False, file_column_name: str = None, make_sdf: bool = True):

        if not os.path.exists(self.file_path_xyz) or overwrite:
            mat = scipy.io.loadmat(os.path.join(self.data_directory, self.download_info["download_file_name"]))
            coulomb_mat = mat["X"]
            graph_len = [int(np.around(np.sum(np.diagonal(x) > 0))) for x in coulomb_mat]
            proton_inv_dist = [coulomb_matrix_to_inverse_distance_proton(x[:i, :i], unit_conversion=0.529177210903) for
                               x, i in zip(coulomb_mat, graph_len)]
            proton = [x[1] for x in proton_inv_dist]
            inv_dist = [x[0] for x in proton_inv_dist]
            dist = [invert_distance(x) for x in inv_dist]
            pos = [coordinates_from_distance_matrix(x) for x in dist]
            atoms = [[inverse_global_proton_dict[i] for i in x] for x in proton]
            atoms_pos = [[x, y] for x, y in zip(atoms, pos)]
            self.info("Writing XYZ file from coulomb matrix information.")
            write_list_to_xyz_file(self.file_path_xyz, atoms_pos)
        else:
            self.info("Found XYZ file for qm7b already created.")

        if not os.path.exists(self.file_path) or overwrite:
            mat = scipy.io.loadmat(os.path.join(self.data_directory, self.download_info["download_file_name"]))
            labels = mat["T"]
            targets = pd.DataFrame(labels, columns=self.label_names)
            self.info("Writing CSV file of graph labels.")
            targets.to_csv(self.file_path, index=False)
        else:
            self.info("Found CSV file of graph labels.")

        return super(QM7bDataset, self).prepare_data(
            overwrite=overwrite, file_column_name=file_column_name, make_sdf=make_sdf)

import os
import numpy as np
from kgcnn.data.qm import QMDataset
from kgcnn.data.download import DownloadDataset


class QM8Dataset(QMDataset, DownloadDataset):
    r"""Store and process QM8 dataset from `MoleculeNet <https://moleculenet.org/>`__ datasets.

    From `Quantum Machine <http://quantum-machine.org/datasets/>`__ :
    Due to its favorable computational efficiency, time-dependent (TD) density functional theory(DFT) enables
    the prediction of electronic spectra in a high-throughput manner across chemical space. Its predictions,
    however, can be quite inaccurate. We resolve this issue with machine learning models trained on deviations of
    reference second-order approximate coupled-cluster (CC2) singles and doubles spectra from TDDFT counterparts,
    or even from DFT gap. We applied this approach to low-lying singlet-singlet vertical electronic spectra of
    over 20000 synthetically feasible small organic molecules with up to eight CONF atoms. The prediction errors
    decay monotonously as a function of training set size. For a training set of 10000 molecules,
    CC2 excitation energies can be reproduced to within ±0.1 eV for the remaining molecules.
    Analysis of our spectral database via chromophore counting suggests that even higher accuracies can be achieved.
    Based on the evidence collected, we discuss open challenges associated with data-driven modeling of
    high-lying spectra and transition intensities.

    .. note::

        We take the pre-processed dataset from `MoleculeNet <https://moleculenet.org/>`_ .

    References:

        (1) L. Ruddigkeit, R. van Deursen, L. C. Blum, J.-L. Reymond, Enumeration of 166 billion organic small
            molecules in the chemical universe database GDB-17, J. Chem. Inf. Model. 52, 2864–2875, 2012.
        (2) R. Ramakrishnan, M. Hartmann, E. Tapavicza, O. A. von Lilienfeld, Electronic Spectra from TDDFT and
            Machine Learning in Chemical Space, J. Chem. Phys. 143 084111, 2015.

    """

    download_info = {
        "dataset_name": "QM8",
        "data_directory_name": "qm8",
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb8.tar.gz",
        "download_file_name": 'gdb8.tar.gz',
        "unpack_tar": True,
        "unpack_zip": False,
        "unpack_directory_name": "gdb8"
    }

    def __init__(self, reload: bool = False, verbose: int = 10):
        """Initialize QM8 dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        QMDataset.__init__(self, verbose=verbose, dataset_name="QM8")
        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)

        self.label_names = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0",
            "f2-PBE0", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM"
        ]
        self.label_units = ["[?]"]*len(self.label_names)
        self.label_unit_conversion = np.array([1.0] * 14)  # Pick always same units for training
        self.dataset_name = "QM8"
        self.require_prepare_data = False
        self.fits_in_memory = True
        self.verbose = verbose
        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name, self.unpack_directory_name)
        self.file_name = "qm8.csv"

        if not os.path.exists(self.file_path):
            original_name = os.path.join(self.data_directory, "qm8.sdf.csv")
            if os.path.exists(original_name):
                os.rename(original_name, self.file_path)

        if self.require_prepare_data:
            self.prepare_data(overwrite=reload)

        if self.fits_in_memory:
            self.read_in_memory(label_column_name=self.label_names)

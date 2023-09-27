import os
import numpy as np
from ase.db import connect

from kgcnn.data.base import MemoryGraphDataset
from kgcnn.data.download import DownloadDataset


class ISO17Dataset(DownloadDataset, MemoryGraphDataset):
    r"""Dataset 'ISO17' with molecules randomly taken from QM9 dataset [1] with a fixed composition of atoms (C7O2H10).
    They were arranged in different chemically valid structures and is an extension of the isomer MD data used in [2].

    Information below and the dataset itself is copied and downloaded from `<http://quantum-machine.org/datasets/>`_ .
    The database consist of 129 molecules each containing 5000 conformational geometries,
    energies and forces with a resolution of 1 femto-second in the molecular dynamics trajectories.
    The database was generated from molecular dynamics simulations using the Fritz-Haber Institute ab initio simulation
    package (FHI-aims)[3]. The simulations in ISO17 were carried out using the standard quantum chemistry computational
    method  density functional theory (DFT) in the generalized gradient approximation (GGA) with the
    Perdew-Burke-Ernzerhof (PBE) functional[4] and the Tkatchenko-Scheffler (TS) van der Waals correction method [5].
    The dataset is stored in ASE sqlite format with the total energy in eV and forces in eV/Ang.
    Download-url: `<http://quantum-machine.org/datasets/iso17.tar.gz>`_ .

    .. code-block:: python

        from ase.db import connect
        with connect(path_to_db) as conn:
           for row in conn.select(limit=10):
               print(row.toatoms())
               print(row['total_energy'])
               print(row.data['atomic_forces'])

    References:

        (1) R. Ramakrishnan et al. Quantum chemistry structures and properties of 134 kilo molecules.
            Scientific Data, 1. (2014) `<https://www.nature.com/articles/sdata201422>`_ .
        (2) Schütt, K. T. et al. Quantum-chemical insights from deep tensor neural networks.
            Nature Communications, 8, 13890. (2017) `<https://www.nature.com/articles/ncomms13890>`_ .
        (3) Blum, V. et al. Ab Initio Molecular Simulations with Numeric Atom-Centered Orbitals.
            Comput. Phys. Commun. 2009, 180 (11), 2175–2196
            `<https://www.sciencedirect.com/science/article/pii/S0010465509002033>`_ .
        (4) Perdew, J. P. et al. Generalized Gradient Approximation Made Simple. Phys. Rev. Lett. 1996, 77 (18),
            3865–3868 `<https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.77.3865>`_.
        (5) Tkatchenko, A. et al. Accurate Molecular Van Der Waals Interactions from Ground-State Electron Density and
            Free-Atom Reference Data. Phys. Rev. Lett. 2009, 102 (7), 73005
            `<https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.102.073005>`_.
        (6) Schütt, K. T., et al. SchNet: A continuous-filter convolutional neural network for modeling quantum
            interactions. Advances in Neural Information Processing System (accepted). 2017.
            `<https://arxiv.org/abs/1706.08566>`_
    """

    download_info = {
        "dataset_name": "ISO17",
        "data_directory_name": "ISO17",
        "download_url": "http://quantum-machine.org/datasets/iso17.tar.gz",
        "download_file_name": 'iso17.tar.gz',
        "unpack_tar": True,
        "unpack_zip": False,
        "unpack_directory_name": "iso17"
    }

    def __init__(self, reload=False, verbose: int = 10):
        r"""Initialize full :obj:`ISO17Dataset` dataset.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        # Use default base class init()
        self.data_keys = None

        MemoryGraphDataset.__init__(self, dataset_name="ISO17", verbose=verbose)
        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)

        self.file_name = ["reference.db", "reference_eq.db", "test_within.db", "test_other.db", "test_eq.db"]
        self.data_directory = os.path.join(
            self.data_main_dir, self.data_directory_name, self.unpack_directory_name, "iso17")
        self.file_directory = self.data_directory

        if self.fits_in_memory:
            self.read_in_memory()

    def read_in_memory(self):
        r"""Load complete :obj:`ISO17Dataset` data into memory. Additionally, the different train and validation
        properties are set according to `http://quantum-machine.org/datasets/ <http://quantum-machine.org/datasets/>`_.

        The data is partitioned as used in the SchNet paper [6]:

            - reference.db: 80% of steps of 80% of MD trajectories
            - reference_eq.db: equilibrium conformations of those molecules
            - test_within.db: remaining 20% unseen steps of reference trajectories
            - test_other.db: remaining 20% unseen MD trajectories
            - test_eq.db: equilibrium conformations of test trajectories

        Where 'reference.db' and 'reference_eq.db' have 'train' property with index 0, 1, respectively, and
        'test_within', 'test_other', 'test_eq' have 'test' property 0, 1, 2, respectively.
        The original validation geometries are noted by 'valid' property with index 0.
        Use :obj:`get_train_test_indices` for reading out split indices.
        """

        data = {"formula": [], "numbers": [], "symbols": [], "positions": [],
                "atomic_forces": [], "total_energy": [], "id": [], "train": [], "test": []}
        # Add them in the order as given in doc string. 'reference' prefix are training data.
        for db_name, train, test in [("reference.db", 0, None), ("reference_eq.db", 1, None),
                                     ("test_within.db", None, 0), ("test_other.db", None, 1), ("test_eq.db", None, 2)]:
            with connect(os.path.join(self.data_directory, db_name)) as conn:
                for row in conn.select():
                    # print(str(row.toatoms()))  # The ase atoms object.
                    data["numbers"].append(row.numbers)
                    data["id"].append(row.id)
                    data["formula"].append(str(row.toatoms().symbols))
                    data["positions"].append(row.positions)
                    data["symbols"].append(row.symbols)
                    data["total_energy"].append(np.expand_dims(row['total_energy'], axis=-1))
                    data["atomic_forces"].append(row.data['atomic_forces'])
                    data["train"].append(train)
                    data["test"].append(test)

        for name, values in data.items():
            self.assign_property(name, values)

        # Validation indices are for the first part that is 'reference.db' which are at the beginning of the dataset.
        # Can be simply added by index.
        with open(os.path.join(self.data_directory, "validation_ids.txt")) as f:
            valid_indices = [int(x.strip()) for x in f.readlines()]
        for i in valid_indices:
            self[i-1].update({"valid": np.array(0)})
        return self

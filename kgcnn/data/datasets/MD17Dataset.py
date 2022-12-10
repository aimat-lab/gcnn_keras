import os
import numpy as np
from typing import Union
from kgcnn.data.base import MemoryGraphDataset
from kgcnn.data.download import DownloadDataset


class MD17Dataset(DownloadDataset, MemoryGraphDataset):
    r"""Store and process trajectories from the :obj:`MD17Dataset` dataset.
    The dataset contains atomic coordinates of molecular dynamics trajectories, as well as the total energy
    (in kcal/mol) and forces (kcal/mol/Angstrom) on each atom.
    For reference data source, refer to the links `<http://www.sgdml.org/#datasets>`_ or
    `<http://quantum-machine.org/gdml/data/>`_ .

    Which trajectory is downloaded is determined by :obj:`trajectory_name` argument.
    There are two different versions of trajectories, which are a long trajectory on DFT
    level of theory and a short trajectory on coupled cluster level of theory marked in the name by
    'dft' and 'ccsd_t' respectively.

    Overview:

        .. list-table::
            :widths: 20 10 20 10
            :header-rows: 1

            * - Molecule
              - Level of Theory
              - trajectory_name
              - graphs
            * - Aspirin
              - DFT
              - aspirin_dft
              - 211762
            * - Azobenzene
              - DFT
              - azobenzene_dft
              - 99999
            * - Benzene
              - DFT
              - benzene2017_dft
              - 627983
            * - Benzene
              - DFT
              - benzene2018_dft
              - 49863
            * - Ethanol
              - DFT
              - ethanol_dft
              - 555092
            * - Malonaldehyde
              - DFT
              - malonaldehyde_dft
              - 993237
            * - Naphthalene
              - DFT
              - naphthalene_dft
              - 326250
            * - Paracetamol
              - DFT
              - paracetamol_dft
              - 106490
            * - Salicylic
              - DFT
              - salicylic_dft
              - 320231
            * - Toluene
              - DFT
              - toluene_dft
              - 442790
            * - Uracil
              - DFT
              - uracil_dft
              - 133770
            * - Aspirin_ccsd
              - CCSD
              - aspirin_ccsd
              - 1500
            * - Benzene
              - CCSD
              - benzene_ccsd_t
              - 1500
            * - Ethanol
              - CCSD
              - ethanol_ccsd_t
              - 2000
            * - Malonaldehyde
              - CCSD
              - malonaldehyde_ccsd_t
              - 1500
            * - Toluene
              - CCSD
              - toluene_ccsd_t
              - 1501

    It is recommended to use the given train-test splits. Only the requested trajectory is downloaded.

    """
    datasets_download_info = {
        "CG-CG": {"download_file_name": "CG-CG.npz"},
        "aspirin_dft": {"download_file_name": "aspirin_dft.npz"},
        "azobenzene_dft": {"download_file_name": "azobenzene_dft.npz"},
        "benzene2017_dft": {"download_file_name": "benzene2017_dft.npz"},
        "benzene2018_dft": {"download_file_name": "benzene2018_dft.npz"},
        "ethanol_dft": {"download_file_name": "ethanol_dft.npz"},
        "malonaldehyde_dft": {"download_file_name": "malonaldehyde_dft.npz"},
        "naphthalene_dft": {"download_file_name": "naphthalene_dft.npz"},
        "paracetamol_dft": {"download_file_name": "paracetamol_dft.npz"},
        "salicylic_dft": {"download_file_name": "salicylic_dft.npz"},
        "toluene_dft": {"download_file_name": "toluene_dft.npz"},
        "uracil_dft": {"download_file_name": "uracil_dft.npz"},
        "aspirin_ccsd": {"download_file_name": "aspirin_ccsd.zip", "unpack_zip": True,
                         "unpack_directory_name": "aspirin_ccsd"},
        "benzene_ccsd_t": {"download_file_name": "benzene_ccsd_t.zip", "unpack_zip": True,
                         "unpack_directory_name": "benzene_ccsd_t"},
        "ethanol_ccsd_t": {"download_file_name": "ethanol_ccsd_t.zip", "unpack_zip": True,
                         "unpack_directory_name": "ethanol_ccsd_t"},
        "malonaldehyde_ccsd_t": {"download_file_name": "malonaldehyde_ccsd_t.zip", "unpack_zip": True,
                                 "unpack_directory_name": "malonaldehyde_ccsd_t"},
        "toluene_ccsd_t": {"download_file_name": "toluene_ccsd_t.zip", "unpack_zip": True,
                           "unpack_directory_name": "toluene_ccsd_t"},
    }

    def __init__(self, trajectory_name: str = None, reload=False, verbose=10):
        """Initialize MD17Dataset dataset.

        Args:
            trajectory_name (str): Name of a trajectory or molecule.
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 60=silent. Default is 10.
        """
        self.data_keys = None
        self.trajectory_name = trajectory_name
        MemoryGraphDataset.__init__(self, dataset_name="MD17", verbose=verbose)

        # Prepare download
        if trajectory_name in self.datasets_download_info:
            self.download_info = self.datasets_download_info[trajectory_name]
            self.download_info.update({
                "download_url": "http://quantum-machine.org/gdml/data/npz/%s" % self.download_info[
                    "download_file_name"]})
        else:
            raise ValueError(
                "Can not resolve '%s' trajectory. Choose: %s." % (
                    trajectory_name, list(self.datasets_download_info.keys())))

        DownloadDataset.__init__(self, dataset_name="MD17", data_directory_name="MD17", **self.download_info,
                                 reload=reload, verbose=verbose)

        self.file_name = str(self.download_file_name) if not self.unpack_zip else [
            os.path.splitext(self.download_file_name)[0] + "-train.npz",
            os.path.splitext(self.download_file_name)[0] + "-test.npz"
        ]
        if self.unpack_directory_name is None:
            self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name)
        else:
            self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name, self.unpack_directory_name)

        self.dataset_name = self.dataset_name + "_" + self.trajectory_name
        if self.fits_in_memory:
            self.read_in_memory()

    def _get_trajectory_from_npz(self, file_path: Union[str, list, tuple] = None):
        # If a filepath is given.
        if file_path is not None:
            if isinstance(file_path, (list, tuple)):
                return [np.load(x) for x in file_path]
            return np.load(file_path)

        # Determine filepath from dataset information.
        if isinstance(self.file_name, str):
            file_path = os.path.join(self.data_directory, self.file_name)
            return np.load(file_path)
        elif isinstance(self.file_name, (list, tuple)):
            file_path = [os.path.join(self.data_directory, x) for x in self.file_name]
        else:
            TypeError("Unknown type for file name '%s'." % self.file_name)
        return [np.load(x) for x in file_path]

    def read_in_memory(self):
        """Load a trajectory into memory."""
        data_loaded = self._get_trajectory_from_npz()

        def make_dict_from_data(data, is_split: dict = None):
            out_dict = {}
            data_keys = list(data.keys())
            # note: Could check if all keys are available here.
            for key in ["R", "E", "F"]:
                out_dict.update({key: [np.array(x) for x in data[key]]})
            num_data_points = len(out_dict["R"])
            for key in ["z", 'name', 'type', 'md5', "theory"]:
                value = data[key]
                out_dict.update({key: [np.array(value) for _ in range(num_data_points)]})
            if is_split is not None:
                for key, value in is_split.items():
                    out_dict.update({key: [value for _ in range(num_data_points)]})
            return out_dict

        if isinstance(data_loaded, (list, tuple)):
            split_assignment = [{"train": np.array([1]), "test": None}, {"train": None, "test": np.array([1])}]
            prop_dicts = [make_dict_from_data(x, is_split=split_assignment[i]) for i, x in enumerate(data_loaded)]
            for key_prop in prop_dicts[0].keys():
                # note: use from itertools import chain for multiple splits.
                self.assign_property(key_prop, prop_dicts[0][key_prop] + prop_dicts[1][key_prop])
        else:
            for key_prop, value_prop in make_dict_from_data(data_loaded).items():
                self.assign_property(key_prop, value_prop)

        return self

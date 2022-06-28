import os

from kgcnn.data.crystal import CrystalDataset
from kgcnn.data.download import DownloadDataset
from kgcnn.data.utils import load_json_file


class MatBenchDataset2020(CrystalDataset, DownloadDataset):
    """Downloader for DeepChem MoleculeNetDataset 2018 class.

    """
    datasets_download_info = {
        "matbench_steels": {"dataset_name": "matbench_steels",
                            "download_file_name": 'matbench_steels.json.gz',
                            "data_directory_name": "matbench_steels", "extract_gz": True,
                            "extract_file_name": 'matbench_steels.json'},
        "matbench_jdft2d": {"dataset_name": "matbench_jdft2d",
                            "download_file_name": 'matbench_jdft2d.json.gz',
                            "data_directory_name": "matbench_jdft2d", "extract_gz": True,
                            "extract_file_name": 'matbench_jdft2d.json'},
        "matbench_phonons": {"dataset_name": "matbench_phonons",
                            "download_file_name": 'matbench_phonons.json.gz',
                            "data_directory_name": "matbench_phonons", "extract_gz": True,
                            "extract_file_name": 'matbench_phonons.json'},

    }
    datasets_prepare_data_info = {
        "matbench_steels": {},
        "matbench_jdft2d": {},
        "matbench_phonons": {}
    }
    datasets_read_in_memory_info = {
        "matbench_steels": {},
        "matbench_jdft2d": {},
        "matbench_phonons": {}
    }

    def __init__(self, dataset_name: str, reload: bool = False, verbose: int = 1):
        """Initialize a `GraphTUDataset` instance from string identifier.

        Args:
            dataset_name (str): Name of a dataset.
            reload (bool): Download the dataset again and prepare data on disk.
            verbose (int): Print progress or info for processing, where 0 is silent. Default is 1.
        """
        if not isinstance(dataset_name, str):
            raise ValueError("Please provide string identifier for TUDataset.")

        CrystalDataset.__init__(self, verbose=verbose, dataset_name=dataset_name)

        # Prepare download
        if dataset_name in self.datasets_download_info:
            self.download_info = self.datasets_download_info[dataset_name]
            self.download_info.update({"download_url": "https://ml.materialsproject.org/projects/" +
                                                       self.download_info["download_file_name"]})
        else:
            raise ValueError("ERROR: Can not resolve %s as a MatBench dataset. Pick " % dataset_name,
                             self.datsets_download_info.keys(),
                             "For new dataset, add to `datasets_download_info` list manually.")

        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)

        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name)
        self.file_name = self.download_file_name if self.extract_file_name is None else self.extract_file_name
        self.dataset_name = dataset_name
        self.require_prepare_data = True
        self.fits_in_memory = False

        if self.require_prepare_data:
            self.prepare_data(overwrite=reload, **self.datasets_prepare_data_info[self.dataset_name])
        if self.fits_in_memory:
            self.read_in_memory(**self.datasets_read_in_memory_info[self.dataset_name])

    def prepare_data(self, cif_column_name: str = None, overwrite: bool = False):
        self.data = load_json_file(self.file_path)


data = MatBenchDataset2020("matbench_steels")
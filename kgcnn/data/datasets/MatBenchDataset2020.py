import os
import pandas as pd

from kgcnn.data.crystal import CrystalDataset
from kgcnn.data.download import DownloadDataset
from kgcnn.data.utils import load_json_file, save_json_file


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
        "matbench_expt_gap": {"dataset_name": "matbench_expt_gap",
                              "download_file_name": 'matbench_expt_gap.json.gz',
                              "data_directory_name": "matbench_expt_gap", "extract_gz": True,
                              "extract_file_name": 'matbench_expt_gap.json'},
        "matbench_dielectric": {"dataset_name": "matbench_dielectric",
                                "download_file_name": 'matbench_dielectric.json.gz',
                                "data_directory_name": "matbench_dielectric", "extract_gz": True,
                                "extract_file_name": 'matbench_dielectric.json'},
        "matbench_expt_is_metal": {"dataset_name": "matbench_expt_is_metal",
                                   "download_file_name": 'matbench_expt_is_metal.json.gz',
                                   "data_directory_name": "matbench_expt_is_metal", "extract_gz": True,
                                   "extract_file_name": 'matbench_expt_is_metal.json'},
        "matbench_glass": {"dataset_name": "matbench_glass",
                           "download_file_name": 'matbench_glass.json.gz',
                           "data_directory_name": "matbench_glass", "extract_gz": True,
                           "extract_file_name": 'matbench_glass.json'},
        "matbench_log_gvrh": {"dataset_name": "matbench_log_gvrh",
                              "download_file_name": 'matbench_log_gvrh.json.gz',
                              "data_directory_name": "matbench_log_gvrh", "extract_gz": True,
                              "extract_file_name": 'matbench_log_gvrh.json'},
        "matbench_log_kvrh": {"dataset_name": "matbench_log_kvrh",
                              "download_file_name": 'matbench_log_kvrh.json.gz',
                              "data_directory_name": "matbench_log_kvrh", "extract_gz": True,
                              "extract_file_name": 'matbench_log_kvrh.json'},
        "matbench_perovskites": {"dataset_name": "matbench_perovskites",
                                 "download_file_name": 'matbench_perovskites.json.gz',
                                 "data_directory_name": "matbench_perovskites", "extract_gz": True,
                                 "extract_file_name": 'matbench_perovskites.json'},
        "matbench_mp_gap": {"dataset_name": "matbench_mp_gap",
                            "download_file_name": 'matbench_mp_gap.json.gz',
                            "data_directory_name": "matbench_mp_gap", "extract_gz": True,
                            "extract_file_name": 'matbench_mp_gap.json'},
        "matbench_mp_is_metal": {"dataset_name": "matbench_mp_is_metal",
                                 "download_file_name": 'matbench_mp_is_metal.json.gz',
                                 "data_directory_name": "matbench_mp_is_metal", "extract_gz": True,
                                 "extract_file_name": 'matbench_mp_is_metal.json'},
        "matbench_mp_e_form": {"dataset_name": "matbench_mp_e_form",
                               "download_file_name": 'matbench_mp_e_form.json.gz',
                               "data_directory_name": "matbench_mp_e_form", "extract_gz": True,
                               "extract_file_name": 'matbench_mp_e_form.json'},

    }
    datasets_prepare_data_info = {
        "matbench_steels": {},
        "matbench_jdft2d": {},
        "matbench_phonons": {},
        "matbench_expt_gap": {},
        "matbench_dielectric": {},
        "matbench_expt_is_metal": {},
        "matbench_glass": {},
        "matbench_log_gvrh": {},
        "matbench_log_kvrh": {},
        "matbench_perovskites": {},
        "matbench_mp_gap": {},
        "matbench_mp_is_metal": {"cif_column_name": "structure"},
        "matbench_mp_e_form": {},
    }
    datasets_read_in_memory_info = {
        "matbench_steels": {},
        "matbench_jdft2d": {},
        "matbench_phonons": {},
        "matbench_expt_gap": {},
        "matbench_dielectric": {},
        "matbench_expt_is_metal": {},
        "matbench_glass": {},
        "matbench_log_gvrh": {},
        "matbench_log_kvrh": {},
        "matbench_perovskites": {},
        "matbench_mp_gap": {},
        "matbench_mp_is_metal": {},
        "matbench_mp_e_form": {},
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

        file_name_base = os.path.splitext(self.file_name)[0]
        if all([os.path.exists(os.path.join(self.data_directory, "%s.pymatgen.json" % file_name_base)),
               os.path.exists(os.path.join(self.data_directory, "%s.csv" % file_name_base)),
               not overwrite]):
            return self

        data = load_json_file(self.file_path)
        # print(data.keys())
        # print(data["columns"])
        data_columns = data["columns"]
        index_structure = 0
        for i, col in enumerate(data_columns):
            if col == cif_column_name:
                index_structure = i
                break
        py_mat_list = [x[index_structure] for x in data["data"]]
        save_json_file(py_mat_list, os.path.join(self.data_directory, "%s.pymatgen.json" % file_name_base))
        df_dict = {"index": data["index"]}
        for i, col in enumerate(data_columns):
            if i != index_structure:
                df_dict[col] = [x[i] for x in data["data"]]
        df = pd.DataFrame(df_dict)
        df.to_csv(os.path.join(self.data_directory, "%s.csv" % file_name_base))
        return self


dataset = MatBenchDataset2020("matbench_mp_is_metal")
print()

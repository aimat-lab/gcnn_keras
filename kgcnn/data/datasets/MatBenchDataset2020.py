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
        "matbench_steels": {"cif_column_name": "composition"},
        "matbench_jdft2d": {"cif_column_name": "structure"},
        "matbench_phonons": {"cif_column_name": "structure"},
        "matbench_expt_gap": {"cif_column_name": "composition"},
        "matbench_dielectric": {"cif_column_name": "structure"},
        "matbench_expt_is_metal": {"cif_column_name": "composition"},
        "matbench_glass": {"cif_column_name": "composition"},
        "matbench_log_gvrh": {"cif_column_name": "structure"},
        "matbench_log_kvrh": {"cif_column_name": "structure"},
        "matbench_perovskites": {"cif_column_name": "structure"},
        "matbench_mp_gap": {"cif_column_name": "structure"},
        "matbench_mp_is_metal": {"cif_column_name": "structure"},
        "matbench_mp_e_form": {"cif_column_name": "structure"},
    }
    datasets_read_in_memory_info = {
        "matbench_steels": {"label_column_name": "yield strength"},
        "matbench_jdft2d": {"label_column_name": "exfoliation_en"},
        "matbench_phonons": {"label_column_name": "last phdos peak"},
        "matbench_expt_gap": {"label_column_name": "gap expt"},
        "matbench_dielectric": {"label_column_name": "n"},
        "matbench_expt_is_metal": {"label_column_name": "is_metal"},
        "matbench_glass": {"label_column_name": "gfa"},
        "matbench_log_gvrh": {"label_column_name": "log10(G_VRH)"},
        "matbench_log_kvrh": {"label_column_name": "log10(K_VRH)"},
        "matbench_perovskites": {"label_column_name": "e_form"},
        "matbench_mp_gap": {"label_column_name": "gap pbe"},
        "matbench_mp_is_metal": {"label_column_name": "is_metal"},
        "matbench_mp_e_form": {"label_column_name": "e_form"},
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
        file_name_download = self.download_file_name if self.extract_file_name is None else self.extract_file_name
        self.file_name = "%s.csv" % os.path.splitext(file_name_download)[0]
        self.dataset_name = dataset_name
        self.require_prepare_data = True
        self.fits_in_memory = True

        if self.require_prepare_data:
            self.prepare_data(overwrite=reload, **self.datasets_prepare_data_info[self.dataset_name])
        if self.fits_in_memory:
            self.read_in_memory(**self.datasets_read_in_memory_info[self.dataset_name])

    def prepare_data(self, cif_column_name: str = None, overwrite: bool = False):

        file_name_download = self.download_file_name if self.extract_file_name is None else self.extract_file_name
        # file_name_base = os.path.splitext(self.file_name)[0]

        if all([os.path.exists(os.path.join(self.data_directory, self._get_pymatgen_file_name())),
               os.path.exists(os.path.join(self.data_directory, self.file_name)),
               not overwrite]):
            self.info("Found %s of structures." % self._get_pymatgen_file_name())
            return self

        self.info("Load dataset '%s' to memory..." % self.dataset_name)
        data = load_json_file(os.path.join(self.data_directory, file_name_download))

        self.info("Process database with %s and columns %s" % (data.keys(), data["columns"]))
        data_columns = data["columns"]
        index_structure = 0
        for i, col in enumerate(data_columns):
            if col == cif_column_name:
                index_structure = i
                break
        py_mat_list = [x[index_structure] for x in data["data"]]

        self.info("Write structures or compositions '%s' to file." % self._get_pymatgen_file_name())
        save_json_file(py_mat_list, os.path.join(self.data_directory, self._get_pymatgen_file_name()))
        df_dict = {"index": data["index"]}
        for i, col in enumerate(data_columns):
            if i != index_structure:
                df_dict[col] = [x[i] for x in data["data"]]
        df = pd.DataFrame(df_dict)

        self.info("Write dataset table '%s' to file." % self.file_name)
        df.to_csv(self.file_path)
        return self


# dataset = MatBenchDataset2020("matbench_mp_e_form", reload=False)
# print("Okay")

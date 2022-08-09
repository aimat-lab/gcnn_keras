import os

from kgcnn.data.moleculenet import MoleculeNetDataset
from kgcnn.data.download import DownloadDataset


class MoleculeNetDataset2018(MoleculeNetDataset, DownloadDataset):
    """Downloader for DeepChem MoleculeNetDataset 2018 class.

    """
    datsets_download_info = {
        "ESOL": {"dataset_name": "ESOL", "download_file_name": 'delaney-processed.csv', "data_directory_name": "ESOL"},
        "FreeSolv": {"dataset_name": "FreeSolv", "data_directory_name": "FreeSolv", "download_file_name": 'SAMPL.csv'},
        "Lipop": {"dataset_name": "Lipop", "data_directory_name": "Lipop", "download_file_name": 'Lipophilicity.csv'},
        "PCBA": {"dataset_name": "PCBA", "data_directory_name": "PCBA", "download_file_name": 'pcba.csv.gz',
                 "extract_gz": True, "extract_file_name": 'pcba.csv'},
        "MUV": {"dataset_name": "MUV", "data_directory_name": "MUV", "download_file_name": 'muv.csv.gz',
                "extract_gz": True, "extract_file_name": 'muv.csv'},
        "HIV": {"dataset_name": "HIV", "data_directory_name": "HIV", "download_file_name": 'HIV.csv'},
        "BACE": {"dataset_name": "BACE", "data_directory_name": "BACE", "download_file_name": 'bace.csv'},
        "BBBP": {"dataset_name": "BBBP", "data_directory_name": "BBBP", "download_file_name": 'BBBP.csv'},
        "Tox21": {"dataset_name": "Tox21", "data_directory_name": "Tox21", "download_file_name": 'tox21.csv.gz',
                  "extract_gz": True, "extract_file_name": 'tox21.csv'},
        "ToxCast": {"dataset_name": "ToxCast", "data_directory_name": "ToxCast",
                    "download_file_name": 'toxcast_data.csv.gz', "extract_gz": True,
                    "extract_file_name": 'toxcast_data.csv'},
        "SIDER": {"dataset_name": "SIDER", "data_directory_name": "SIDER", "download_file_name": 'sider.csv.gz',
                  "extract_gz": True, "extract_file_name": 'sider.csv'},
        "ClinTox": {"dataset_name": "ClinTox", "data_directory_name": "ClinTox", "download_file_name": 'clintox.csv.gz',
                    "extract_gz": True, "extract_file_name": 'clintox.csv'},
    }
    datasets_prepare_data_info = {
        "ESOL": {"make_conformers": True, "add_hydrogen": True},
        "FreeSolv": {"make_conformers": True, "add_hydrogen": True},
        "Lipop": {"make_conformers": True, "add_hydrogen": True},
        "PCBA": {"make_conformers": True, "add_hydrogen": True},
        "MUV": {"make_conformers": True, "add_hydrogen": True},
        "HIV": {"make_conformers": True, "add_hydrogen": True},
        "BACE": {"make_conformers": True, "add_hydrogen": True, "smiles_column_name": "mol"},
        "BBBP": {"make_conformers": True, "add_hydrogen": True, "smiles_column_name": "smiles"},
        "Tox21": {"make_conformers": True, "add_hydrogen": True, "smiles_column_name": "smiles"},
        "ToxCast": {"make_conformers": True, "add_hydrogen": True, "smiles_column_name": "smiles"},
        "SIDER": {"make_conformers": True, "add_hydrogen": True, "smiles_column_name": "smiles"},
        "ClinTox": {"make_conformers": True, "add_hydrogen": True, "smiles_column_name": "smiles"}
    }
    datasets_read_in_memory_info = {
        "ESOL": {"add_hydrogen": False, "has_conformers": True,
                 "label_column_name": "measured log solubility in mols per litre"},
        "FreeSolv": {"add_hydrogen": False, "has_conformers": True, "label_column_name": "expt"},
        "Lipop": {"add_hydrogen": False, "has_conformers": True, "label_column_name": "exp"},
        "PCBA": {"add_hydrogen": False, "has_conformers": False,  "label_column_name": slice(0, 128)},
        "MUV": {"add_hydrogen": False, "has_conformers": True,  "label_column_name": slice(0, 17)},
        "HIV": {"add_hydrogen": False,"has_conformers": True, "label_column_name": "HIV_active"},
        "BACE": {"add_hydrogen": False, "has_conformers": True, "label_column_name": "Class"},
        "BBBP": { "add_hydrogen": False, "has_conformers": True, "label_column_name": "p_np"},
        "Tox21": {"add_hydrogen": False, "has_conformers": True, "label_column_name": slice(0, 12)},
        "ToxCast": {"add_hydrogen": False, "has_conformers": True, "label_column_name": slice(1, 618)},
        "SIDER": {"add_hydrogen": False, "has_conformers": True, "label_column_name": slice(1, 28)},
        "ClinTox": {"add_hydrogen": False, "has_conformers": True,  "label_column_name": [1, 2]}
    }

    def __init__(self, dataset_name: str, reload: bool = False, verbose: int = 1):
        """Initialize a `MoleculeNetDataset2018` instance from string identifier.

        Args:
            dataset_name (str): Name of a dataset.
            reload (bool): Download the dataset again and prepare data on disk.
            verbose (int): Print progress or info for processing, where 0 is silent. Default is 1.
        """
        if not isinstance(dataset_name, str):
            raise ValueError("Please provide string identifier for TUDataset.")

        MoleculeNetDataset.__init__(self, verbose=verbose, dataset_name=dataset_name)

        # Prepare download
        if dataset_name in self.datsets_download_info:
            self.download_info = self.datsets_download_info[dataset_name]
            self.download_info.update({"download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/" +
                                                       self.download_info["download_file_name"]})
        else:
            raise ValueError("ERROR:kgcnn: Can not resolve %s as a Molecule. Pick " % dataset_name,
                             self.datsets_download_info.keys(),
                             "For new dataset, add to `datsets_download_info` list manually.")

        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)

        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name)
        self.file_name = self.download_file_name if self.extract_file_name is None else self.extract_file_name
        self.dataset_name = dataset_name
        self.require_prepare_data = True
        self.fits_in_memory = True

        if self.require_prepare_data:
            self.prepare_data(overwrite=reload, **self.datasets_prepare_data_info[self.dataset_name])
        if self.fits_in_memory:
            self.read_in_memory(**self.datasets_read_in_memory_info[self.dataset_name])


# data = MoleculeNetDataset2018("ESOL", reload=True).set_attributes()
# data = MoleculeNetDataset2018("PCBA", reload=False).set_attributes()
# data = MoleculeNetDataset2018("ClinTox", reload=True).set_attributes()
# data = MoleculeNetDataset2018("Tox21", reload=True).set_attributes()
# data = MoleculeNetDataset2018("HIV", reload=True).set_attributes()

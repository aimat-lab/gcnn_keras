import os

from kgcnn.data.moleculenet import MoleculeNetDataset
from kgcnn.data.base import DownloadDataset


class MoleculeNetDataset2018(MoleculeNetDataset, DownloadDataset):
    """Downloader for MoleculeNetDataset 2018 class.

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
        "PCBA": {"make_conformers": False, "add_hydrogen": False},
        "MUV": {"make_conformers": False, "add_hydrogen": False},
        "HIV": {"make_conformers": False, "add_hydrogen": False},
        "BACE": {"make_conformers": False, "add_hydrogen": False, "smiles_column_name": "mol"},
        "BBBP": {"make_conformers": False, "add_hydrogen": False, "smiles_column_name": "smiles"},
        "Tox21": {"make_conformers": False, "add_hydrogen": False, "smiles_column_name": "smiles"},
        "ToxCast": {"make_conformers": False, "add_hydrogen": False, "smiles_column_name": "smiles"},
        "SIDER": {"make_conformers": False, "add_hydrogen": False, "smiles_column_name": "smiles"},
        "ClinTox": {"make_conformers": False, "add_hydrogen": False, "smiles_column_name": "smiles"}
    }
    datasets_read_in_memory_info = {
        "ESOL": {"add_hydrogen": False, "has_conformers": True,
                 "label_column_name": "measured log solubility in mols per litre"},
        "FreeSolv": {"has_conformers": True, "add_hydrogen": False, "label_column_name": "expt"},
        "Lipop": {"add_hydrogen": False, "has_conformers": True, "label_column_name": "exp"},
        "PCBA": {"has_conformers": False, "add_hydrogen": False, "label_column_name": slice(0, 128)},
        "MUV": {"has_conformers": False, "add_hydrogen": False, "label_column_name": slice(0, 17)},
        "HIV": {"has_conformers": False, "add_hydrogen": False, "label_column_name": "HIV_active"},
        "BACE": {"has_conformers": False, "add_hydrogen": False, "label_column_name": "Class"},
        "BBBP": {"has_conformers": False, "add_hydrogen": False, "label_column_name": "p_np"},
        "Tox21": {"has_conformers": False, "add_hydrogen": False, "label_column_name": slice(0, 12)},
        "ToxCast": {"has_conformers": False, "add_hydrogen": False, "label_column_name": slice(1, 618)},
        "SIDER": {"has_conformers": False, "add_hydrogen": False, "label_column_name": slice(1, 28)},
        "ClinTox": {"has_conformers": False, "add_hydrogen": False, "label_column_name": slice(1, 3)}
    }

    def __init__(self, dataset_name: str, reload: bool = False, verbose: int = 1):
        """Initialize a `GraphTUDataset` instance from string identifier.

        Args:
            dataset_name (str): Name of a dataset.
            reload (bool): Download the dataset again and prepare data on disk.
            verbose (int): Print progress or info for processing, where 0 is silent. Default is 1.
        """
        if not isinstance(dataset_name, str):
            raise ValueError("ERROR:kgcnn: Please provide string identifier for TUDataset.")

        MoleculeNetDataset.__init__(self, verbose=verbose)

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

        if self.require_prepare_data:
            self.prepare_data(overwrite=reload, verbose=verbose)
        if self.mol_filename:
            self.read_in_memory(verbose=verbose)

    def prepare_data(self, file_name: str = None, data_directory: str = None, dataset_name: str = None,
                     overwrite: bool = False, verbose: int = 1, smiles_column_name: str = "smiles",
                     make_conformers: bool = None, add_hydrogen: bool = None, **kwargs):
        r"""Pre-computation of molecular structure.

        Args:
            file_name (str): Filename for reading into memory. Default is None.
            data_directory (str): Full path to directory containing all files. Default is None.
            dataset_name (str): Name of the dataset. Default is None.
            overwrite (bool): Overwrite existing database mol-json file. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
            smiles_column_name (str): Column name where smiles are given in csv-file. Default is "smiles".
            add_hydrogen (bool): Whether to add H after smile translation. Default is True.
            make_conformers (bool): Whether to make conformers. Default is True.

        Returns:
            self
        """
        prepare_info = {"file_name": file_name, "data_directory": data_directory, "dataset_name": dataset_name,
                        "overwrite": overwrite, "verbose": verbose, "smiles_column_name": smiles_column_name,
                        "add_hydrogen": add_hydrogen, "make_conformers": make_conformers
                        }
        prepare_info.update(self.datasets_prepare_data_info[self.dataset_name])

        if add_hydrogen is not None:
            prepare_info["add_hydrogen"] = add_hydrogen
        if make_conformers is not None:
            make_conformers["make_conformers"] = make_conformers

        return super(MoleculeNetDataset2018, self).prepare_data(**prepare_info)

    def read_in_memory(self, file_name: str = None, data_directory: str = None, dataset_name: str = None,
                       has_conformers: bool = None, label_column_name: str = None,
                       add_hydrogen: bool = None, verbose: int = 1):
        r"""Load list of molecules from json-file named in :obj:`MoleculeNetDataset.mol_filename` into memory. And
        already extract basic graph information. No further attributes are computed as default.

        Args:
            file_name (str): Filename for reading into memory. Default is None.
            data_directory (str): Full path to directory containing all files. Default is None.
            dataset_name (str): Name of the dataset. Default is None.
            has_conformers (bool): If molecules have 3D coordinates pre-computed.
            label_column_name (str): Column name where labels are given in csv-file. Default is None.
            add_hydrogen (bool): Whether to add H after smile translation.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.

        Returns:
            self
        """
        read_in_memory_info = {"file_name": file_name, "data_directory": data_directory, "dataset_name": dataset_name,
                               "verbose": verbose, "label_column_name": label_column_name,
                               "add_hydrogen": add_hydrogen, "has_conformers": has_conformers
                               }
        read_in_memory_info.update(self.datasets_read_in_memory_info[self.dataset_name])

        if add_hydrogen is not None:
            read_in_memory_info["add_hydrogen"] = add_hydrogen
        if has_conformers is not None:
            read_in_memory_info["has_conformers"] = has_conformers
        if label_column_name is not None:
            read_in_memory_info["label_column_name"] = label_column_name

        return super(MoleculeNetDataset2018, self).read_in_memory(**read_in_memory_info)

# data = MoleculeNetDataset2018("ESOL", reload=True).set_attributes()

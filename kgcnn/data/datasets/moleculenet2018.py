import os

from kgcnn.data.moleculenet import MoleculeNetDataset
from kgcnn.data.base import DownloadDataset


class MoleculeNetDataset2018(MoleculeNetDataset, DownloadDataset):
    """Downloader for MoleculeNetDataset 2018 class.

    """
    datsets_download_info = {
        "Lipop": {
            "dataset_name": "Lipop",
            "data_directory_name": "Lipop",
            "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
            "download_file_name": 'Lipophilicity.csv'
        },
        "ESOL": {
            "dataset_name": "ESOL",
            "data_directory_name": "ESOL",
            "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
            "download_file_name": 'delaney-processed.csv'
        },
        "FreeSolv": {
            "dataset_name": "FreeSolv",
            "data_directory_name": "FreeSolv",
            "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
            "download_file_name": 'SAMPL.csv'
        },
        "PCBA": {
            "dataset_name": "PCBA",
            "data_directory_name": "PCBA",
            "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pcba.csv.gz",
            "download_file_name": 'pcba.csv.gz',
            "extract_gz": True,
            "extract_file_name": 'pcba.csv'
        }

    }
    datasets_prepare_data_info= {
        "Lipop": {
            "make_conformers": True,
            "add_hydrogen": True
        },
        "ESOL": {
            "make_conformers": True,
            "add_hydrogen": True
        },
        "FreeSolv": {
            "make_conformers": True,
            "add_hydrogen": True
        },
        "PCBA": {
            "make_conformers": False,
            "add_hydrogen": False
        }
    }
    datasets_read_in_memory_info = {
        "Lipop": {
            "label_column_name": "exp",
            "add_hydrogen": False,
            "has_conformers": True
        },
        "ESOL": {
            "label_column_name": "measured log solubility in mols per litre",
            "add_hydrogen": False,
            "has_conformers": True
        },
        "FreeSolv": {
            "label_column_name": "expt",
            "add_hydrogen": False,
            "has_conformers": True
        },
        "PCBA": {
            "label_column_name": slice(0, 128),
            "add_hydrogen": False,
            "has_conformers": False
        }
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
        else:
            raise ValueError("ERROR:kgcnn: Can not resolve %s as a Molecule." % dataset_name,
                             "Add to `datset_download_info` list manually.")

        DownloadDataset.__init__(self, **self.download_info, reload=reload, verbose=verbose)

        self.data_directory = os.path.join(self.data_main_dir, self.data_directory_name)
        self.file_name = self.download_file_name
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


data = MoleculeNetDataset2018("PCBA")
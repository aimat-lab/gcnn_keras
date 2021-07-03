import numpy as np
import os
import requests
# import scipy.sparse as sp
# import pickle
# import shutil
import tarfile
import zipfile


class GraphDatasetBase:
    """Base layer for datasets. Provides functions for download and unzip of the data.
    Dataset-specific functions like prepare_data() or read_in_memory() must be implemented in subclasses.
    Information about the dataset can be set with class properties.
    Each dataset should implement a get_graph() method which actually makes the graph with specific settings.

    """

    data_main_dir = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets")
    kgcnn_dataset_name = None
    data_directory = None
    download_url = None
    file_name = None
    unpack_tar = False
    unpack_zip = False
    unpack_directory = None
    fits_in_memory = False
    process_dataset = False

    def __init__(self, reload=False, verbose=1):
        """Base initialization function for downloading and extracting the data.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        # Properties that could or should be set by read_in_memory() and get_graph() if memory is not an issue.
        self.data = None
        self.nodes = None
        self.nodes_degree = None
        self.edges = None
        self.labels_graph = None
        self.labels_node = None
        self.labels_edge = None
        self.edge_indices = None
        self.graph_state = None
        self.graph_adjacency = None
        self.atoms = None
        self.coordinates = None

        # Default functions to load a dataset.
        if self.data_directory is not None:
            self.setup_dataset_main(self.data_main_dir, verbose=verbose)
            self.setup_dataset_dir(self.data_main_dir, self.data_directory, verbose=verbose)

        if self.download_url is not None:
            self.download_database(os.path.join(self.data_main_dir, self.data_directory), self.download_url,
                                   self.file_name, overwrite=reload, verbose=verbose)

        if self.unpack_tar:
            self.unpack_tar_file(os.path.join(self.data_main_dir, self.data_directory), self.file_name,
                                 self.unpack_directory, overwrite=reload, verbose=verbose)

        if self.unpack_zip:
            self.unpack_zip_file(os.path.join(self.data_main_dir, self.data_directory), self.file_name,
                                 self.unpack_directory, overwrite=reload, verbose=verbose)

        if self.process_dataset:
            # Used if a standard processing of the data has to be done.
            self.prepare_data(overwrite=reload, verbose=verbose)

        if self.fits_in_memory:
            self.read_in_memory()

    @classmethod
    def setup_dataset_main(cls, data_main_dir, verbose=1):
        """Make the main-directory for all datasets to store data.

        Args:
           data_main_dir (str): Path to create directory.
           verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        os.makedirs(data_main_dir, exist_ok=True)
        if verbose > 0:
            print("INFO: Dataset directory located at", data_main_dir)

    @classmethod
    def setup_dataset_dir(cls, data_main_dir, data_directory, verbose=1):
        """Make directory for each dataset.

        Args:
            data_main_dir (str): Path-location of the directory for all datasets.
            data_directory (str): Path of the directory for specific dataset to create.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        if not os.path.exists(os.path.join(data_main_dir, data_directory)):
            os.mkdir(os.path.join(data_main_dir, data_directory))
        else:
            if verbose > 0:
                print("INFO: Dataset directory found... done")

    @classmethod
    def download_database(cls, path, download_url, filename, overwrite=False, verbose=1):
        """Download dataset file.

        Args:
            path (str): Target filepath to store file (without filename).
            download_url (str): String of the download url to catch database from.
            filename (str): Name the dataset is downloaded to.
            overwrite (bool): Overwrite existing database. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.

        Returns:
            os.path: Filepath of downloaded file.
        """
        if os.path.exists(os.path.join(path, filename)) is False or overwrite:
            if verbose > 0:
                print("INFO: Downloading dataset... ", end='', flush=True)
            r = requests.get(download_url, allow_redirects=True)
            open(os.path.join(path, filename), 'wb').write(r.content)
            if verbose > 0:
                print("done")
        else:
            if verbose > 0:
                print("INFO: Dataset found... done")
        return os.path.join(path, filename)

    @classmethod
    def unpack_tar_file(cls, path, filename, unpack_directory, overwrite=False, verbose=1):
        """Extract tar-file.

        Args:
            path (str): Filepath where the tar-file to extract is located.
            filename (str): Name of the dataset to extract.
            unpack_directory (str): Directory to extract data to.
            overwrite (bool): Overwrite existing database. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.

        Returns:
            os.path: Filepath of the extracted dataset folder.
        """
        if not os.path.exists(os.path.join(path, unpack_directory)):
            if verbose > 0:
                print("INFO: Creating directory... ", end='', flush=True)
            os.mkdir(os.path.join(path, unpack_directory))
            if verbose > 0:
                print("done")
        else:
            if verbose > 0:
                print("INFO: Directory for extraction exists... done")
            if not overwrite:
                if verbose > 0:
                    print("INFO: Not extracting tar File... stopped")
                return os.path.join(path, unpack_directory)  # Stop extracting here

        if verbose > 0:
            print("INFO: Read tar file... ", end='', flush=True)
        archive = tarfile.open(os.path.join(path, filename), "r")
        # Filelistnames = archive.getnames()
        if verbose > 0:
            print("done")

        if verbose > 0:
            print("INFO: Extracting tar file... ", end='', flush=True)
        archive.extractall(os.path.join(path, unpack_directory))
        if verbose > 0:
            print("done")
        archive.close()

        return os.path.join(path, unpack_directory)

    @classmethod
    def unpack_zip_file(cls, path, filename, unpack_directory, overwrite=False, verbose=1):
        """Extract zip-file.

        Args:
            path (str): Filepath where the zip-file to extract is located.
            filename (str): Name of the dataset to extract.
            unpack_directory (str): Directory to extract data to.
            overwrite (bool): Overwrite existing database. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.

        Returns:
            os.path: Filepath of the extracted dataset folder.
        """
        if os.path.exists(os.path.join(path, unpack_directory)):
            if verbose > 0:
                print("INFO: Directory for extraction exists... done")
            if not overwrite:
                if verbose > 0:
                    print("INFO: Not extracting zip file ... stopped")
                return os.path.join(path, unpack_directory)

        if verbose > 0:
            print("INFO: Read zip file ... ", end='', flush=True)
        archive = zipfile.ZipFile(os.path.join(path, filename), "r")
        # Filelistnames = archive.getnames()
        if verbose > 0:
            print("done")

        if verbose > 0:
            print("INFO: Extracting zip file...", end='', flush=True)
        archive.extractall(os.path.join(path, unpack_directory))
        if verbose > 0:
            print("done")
        archive.close()

        return os.path.join(path, unpack_directory)

    @classmethod
    def read_csv_simple(cls, filepath, delimiter=",", dtype=float):
        """Very simple python-only function to read in a csv-file from file.

        Args:
            filepath (str): Full filepath of csv-file to read in.
            delimiter (str): Delimiter character for separation. Default is ",".
            dtype: Callable type conversion from string. Default is float.

        Returns:
            list: Python list of values. Length of the list equals the number of lines.
        """
        out = []
        open_file = open(filepath, "r")
        for lines in open_file.readlines():
            string_list = lines.strip().split(delimiter)
            values_list = [dtype(x.strip()) for x in string_list]
            out.append(values_list)
        open_file.close()
        return out

    def read_in_memory(self, verbose=1):
        pass

    def get_graph(self):
        pass

    def prepare_data(self, overwrite=False, verbose=1):
        pass



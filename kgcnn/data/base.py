import numpy as np
import os
import requests
import scipy.sparse as sp
import pickle
import shutil
import tarfile
import zipfile


class GraphDatasetBase:

    data_main_dir = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets")
    data_directory = None
    download_url = None
    file_name = None
    unpack_tar = False
    unpack_zip = False
    unpack_directory = None
    fits_in_memory = True
    process_dataset = False

    def __init__(self, reload=False):

        if self.data_directory is not None:
            self.setup_dataset_main(self.data_main_dir)
            self.setup_dataset_dir(self.data_main_dir, self.data_directory)

        if self.download_url is not None:
            self.download_database(os.path.join(self.data_main_dir, self.data_directory), self.download_url,
                                   self.file_name, overwrite=reload)

        if self.unpack_tar:
            self.unpack_tar_file(os.path.join(self.data_main_dir, self.data_directory), self.file_name,
                                 self.unpack_directory,overwrite=reload)

        if self.unpack_zip:
            self.unpack_zip_file(os.path.join(self.data_main_dir, self.data_directory), self.file_name,
                                 self.unpack_directory, overwrite=reload)

        if self.process_dataset:
            self.prepare_data()

        if self.fits_in_memory:
            self.read_in_memory()

    @classmethod
    def setup_dataset_main(cls, data_main_dir):
        os.makedirs(data_main_dir, exist_ok=True)
        print("INFO: Dataset directory located at", data_main_dir)

    @classmethod
    def setup_dataset_dir(cls, data_main_dir, data_directory):
        if not os.path.exists(os.path.join(data_main_dir, data_directory)):
            os.mkdir(os.path.join(data_main_dir, data_directory))
        else:
            print("INFO: Dataset directory found... done")

    @classmethod
    def download_database(cls, path, download_url, filename, overwrite=False):
        """Download dataset file.

        Args:
            path (str): Filepath to file.
            download_url (str): String of the download url to catch database from.
            filename (str): Name the dataset is downloaded to.
            overwrite (bool): overwrite existing database, default:False

        Returns:
            os.path: Filepath of downloaded file.
        """
        if os.path.exists(os.path.join(path, filename)) is False or overwrite:
            print("INFO: Downloading dataset... ", end='', flush=True)
            r = requests.get(download_url, allow_redirects=True)
            open(os.path.join(path, filename), 'wb').write(r.content)
            print("done")
        else:
            print("INFO: Dataset found... done")
        return os.path.join(path, filename)

    @classmethod
    def unpack_tar_file(cls, path, filename, unpack_directory, overwrite=False):
        """Extract tar-file.

        Args:
            path (str): Filepath to file.
            filename (str): Name the dataset is downloaded to.
            unpack_directory (str): Directory to extract data to.
            overwrite: (bool) overwrite existing database, default:False

        Returns:
            os.path: Filepath
        """
        if not os.path.exists(os.path.join(path, unpack_directory)):
            print("INFO: Creating directory... ", end='', flush=True)
            os.mkdir(os.path.join(path, unpack_directory))
            print("done")
        else:
            print("INFO: Directory for extraction exists... done")
            if not overwrite:
                print("INFO: Not extracting tar File... stopped")
                return os.path.join(path, unpack_directory)

        print("INFO: Read tar file... ", end='', flush=True)
        archive = tarfile.open(os.path.join(path, filename), "r")
        # Filelistnames = archive.getnames()
        print("done")

        print("INFO: Extracting tar file... ", end='', flush=True)
        archive.extractall(os.path.join(path, unpack_directory))
        print("done")
        archive.close()

        return os.path.join(path, unpack_directory)

    @classmethod
    def unpack_zip_file(cls, path, filename, unpack_directory, overwrite=False):

        if os.path.exists(os.path.join(path, unpack_directory)):
            print("INFO: Directory for extraction exists... done")
            if not overwrite:
                print("INFO: Not extracting zip file ... stopped")
                return os.path.join(path, unpack_directory)

        print("INFO: Read zip file ... ", end='', flush=True)
        archive = zipfile.ZipFile(os.path.join(path, filename), "r")
        # Filelistnames = archive.getnames()
        print("done")

        print("INFO: Extracting zip file...", end='', flush=True)
        archive.extractall(os.path.join(path, unpack_directory))
        print("done")
        archive.close()

        return os.path.join(path, unpack_directory)

    def read_in_memory(self):
        pass

    def get_graph(self):
        pass

    def prepare_data(self):
        pass
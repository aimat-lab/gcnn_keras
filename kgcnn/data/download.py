import gzip
import os
import shutil
import tarfile
import zipfile

import requests


class DownloadDataset:
    r"""Base layer for datasets. Provides class-method functions for download and unzip of the data.
    Dataset-specific functions like prepare_data() must be implemented in subclasses.
    Note that :obj:``DownloadDataset`` uses a main directory located at '~/.kgcnn/datasets' for downloading datasets.
    """

    def __init__(self,
                 dataset_name: str = None,
                 download_file_name: str = None,
                 data_directory_name: str = None,
                 unpack_directory_name: str = None,
                 extract_file_name: str = None,
                 download_url: str = None,
                 unpack_tar: bool = False,
                 unpack_zip: bool = False,
                 extract_gz: bool = False,
                 reload: bool = False,
                 verbose: int = 1,
                 data_main_dir=os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets"),
                 **kwargs):
        r"""Base initialization function for downloading and extracting the data.

        Args:
            dataset_name (str): None.
            download_file_name=None (str): None.
            data_directory_name (str): None.
            unpack_directory_name (str): None.
            extract_file_name (str): None.
            download_url (str): None.
            unpack_tar (bool): False.
            unpack_zip (bool): False.
            extract_gz (bool): False.
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        self.data_main_dir = data_main_dir
        self.dataset_name = dataset_name
        self.download_file_name = download_file_name
        self.data_directory_name = data_directory_name
        self.unpack_directory_name = unpack_directory_name
        self.extract_file_name = extract_file_name
        self.download_url = download_url
        self.unpack_tar = unpack_tar
        self.unpack_zip = unpack_zip
        self.extract_gz = extract_gz
        self.verbose = verbose
        self.download_reload = reload

        # Make the download already in init.
        self.download_dataset_to_disk()

    def download_dataset_to_disk(self):
        """Main download function to store and unpack dataset."""

        # Some datasets do not offer all information or require multiple files.
        if self.verbose > 1:
            print("INFO:kgcnn: Checking and possibly downloading dataset with name %s" % str(self.dataset_name))

        # Default functions to load a dataset.
        if self.data_directory_name is not None:
            self.setup_dataset_main(self.data_main_dir, verbose=self.verbose)
            self.setup_dataset_dir(self.data_main_dir, self.data_directory_name, verbose=self.verbose)

        if self.download_url is not None:
            self.download_database(os.path.join(self.data_main_dir, self.data_directory_name), self.download_url,
                                   self.download_file_name, overwrite=self.download_reload, verbose=self.verbose)

        if self.unpack_tar:
            self.unpack_tar_file(os.path.join(self.data_main_dir, self.data_directory_name), self.download_file_name,
                                 self.unpack_directory_name, overwrite=self.download_reload, verbose=self.verbose)

        if self.unpack_zip:
            self.unpack_zip_file(os.path.join(self.data_main_dir, self.data_directory_name), self.download_file_name,
                                 self.unpack_directory_name, overwrite=self.download_reload, verbose=self.verbose)

        if self.extract_gz:
            self.extract_gz_file(os.path.join(self.data_main_dir, self.data_directory_name), self.download_file_name,
                                 self.extract_file_name, overwrite=self.download_reload, verbose=self.verbose)

    @classmethod
    def setup_dataset_main(cls, data_main_dir, verbose=1):
        """Make the main-directory for all datasets to store data.

        Args:
           data_main_dir (str): Path to create directory.
           verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """
        os.makedirs(data_main_dir, exist_ok=True)
        if verbose > 0:
            print("INFO:kgcnn: Dataset directory located at", data_main_dir)

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
                print("INFO:kgcnn: Dataset directory found... done")

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
                print("INFO:kgcnn: Downloading dataset... ", end='', flush=True)
            r = requests.get(download_url, allow_redirects=True)
            open(os.path.join(path, filename), 'wb').write(r.content)
            if verbose > 0:
                print("done")
        else:
            if verbose > 0:
                print("INFO:kgcnn: Dataset found... done")
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
                print("INFO:kgcnn: Creating directory... ", end='', flush=True)
            os.mkdir(os.path.join(path, unpack_directory))
            if verbose > 0:
                print("done")
        else:
            if verbose > 0:
                print("INFO:kgcnn: Directory for extraction exists... done")
            if not overwrite:
                if verbose > 0:
                    print("INFO:kgcnn: Not extracting tar File... stopped")
                return os.path.join(path, unpack_directory)  # Stop extracting here

        if verbose > 0:
            print("INFO:kgcnn: Read tar file... ", end='', flush=True)
        archive = tarfile.open(os.path.join(path, filename), "r")
        # Filelistnames = archive.getnames()
        if verbose > 0:
            print("done")

        if verbose > 0:
            print("INFO:kgcnn: Extracting tar file... ", end='', flush=True)
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
                print("INFO:kgcnn: Directory for extraction exists... done")
            if not overwrite:
                if verbose > 0:
                    print("INFO:kgcnn: Not extracting zip file ... stopped")
                return os.path.join(path, unpack_directory)

        if verbose > 0:
            print("INFO:kgcnn: Read zip file ... ", end='', flush=True)
        archive = zipfile.ZipFile(os.path.join(path, filename), "r")
        # Filelistnames = archive.getnames()
        if verbose > 0:
            print("done")

        if verbose > 0:
            print("INFO:kgcnn: Extracting zip file...", end='', flush=True)
        archive.extractall(os.path.join(path, unpack_directory))
        if verbose > 0:
            print("done")
        archive.close()

        return os.path.join(path, unpack_directory)

    @classmethod
    def extract_gz_file(cls, path: str, filename: str, out_filename: str = None,
                        overwrite: bool = False, verbose: int = 1):
        """Extract gz-file.

        Args:
            path (str): Filepath where the gz-file to extract is located.
            filename (str): Name of the gz-file to extract.
            out_filename (str): Name of the extracted file.
            overwrite (bool): Overwrite existing database. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.

        Returns:
            os.path: Filepath of the extracted file.
        """
        if out_filename is None:
            out_filename = filename.replace(".gz", "")

        if os.path.exists(os.path.join(path, out_filename)):
            if verbose > 0:
                print("INFO:kgcnn: Extracted file exists... done")
            if not overwrite:
                if verbose > 0:
                    print("INFO:kgcnn: Not extracting gz-file ... stopped")
                return os.path.join(path, out_filename)

        if verbose > 0:
            print("INFO:kgcnn: Extract zgz file ... ", end='', flush=True)

        with gzip.open(os.path.join(path, filename), 'rb') as f_in:
            with open(os.path.join(path, out_filename), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        if verbose > 0:
            print("done")

        return os.path.join(path, out_filename)

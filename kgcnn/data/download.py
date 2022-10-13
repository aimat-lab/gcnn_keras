import gzip
import os
import shutil
import tarfile
import zipfile
import requests
import logging

logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


class DownloadDataset:
    r"""Download class for datasets.

    Provides static-methods and functions for download and unzip of the data.
    They are intentionally kept general and could also be used without this class definition.
    Downloading is handled by :obj:`download_dataset_to_disk` already in :obj:`init` by default.
    Dataset-specific functions like :obj:`prepare_data` must be implemented in subclasses.

    .. note::

        Note that :obj:`DownloadDataset` uses a main directory located at '~/.kgcnn/datasets' for downloading datasets
        as default.

    Classes in :obj:`kgcnn.data.datasets` inherit from this class, but :obj:`DownloadDataset` can also be
    used as a member via composition.

    .. warning::

        Downloads are not checked for safety or malware. Use with caution!

    Example on how to use :obj:`DownloadDataset` standalone:

    .. code-block:: python

        from kgcnn.data.download import DownloadDataset
        download = DownloadDataset(
            download_url="https://github.com/aimat-lab/gcnn_keras/blob/master/README.md",
            data_main_dir="./",
            data_directory_name="",
            download_file_name="README.html",
            reload=True,
            execute_download_dataset_on_init=False
        )
        download.download_dataset_to_disk()
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
                 verbose: int = 10,
                 data_main_dir: str = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets"),
                 execute_download_dataset_on_init: bool = True
                 ):
        r"""Base initialization function for downloading and extracting the data. The arguments to the constructor
        determine what to download and whether to unpack the download. The main function :obj:`download_dataset_to_disk`
        is already called in the constructor.

        Args:
            dataset_name (str): Name of the dataset to download (optional). Default is None.
            download_file_name (str): Name of the file that the source is downloaded to. Default is None.
            data_directory_name (str): Name of the data directory in data_main_dir the file is saved. Default is None.
            unpack_directory_name (str): The name of a new directory in data_directory_name to unpack archive.
                Default is None.
            extract_file_name (str): Name of a gz-file to extract. Default is None.
            download_url (str): Url for file to download. Default is None.
            unpack_tar (bool): Whether to unpack a tar-archive. Default is False.
            unpack_zip (bool): Whether to unpack a zip-archive. Default is False.
            extract_gz (bool): Whether to unpack a gz-archive. Default is False.
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Logging level. Default is 10.
            execute_download_dataset_on_init (bool): Whether to start download on class construction.
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
        self.download_reload = reload
        self.logger_download = module_logger
        self.logger_download.setLevel(verbose)
        self.execute_download_dataset_on_init = bool(execute_download_dataset_on_init)

        # Make the download already in init.
        if self.execute_download_dataset_on_init:
            self.download_dataset_to_disk()

    def download_dataset_to_disk(self):
        """Main download function to run the download and unpack of the dataset. Defined by attributes in self."""

        # Some datasets do not offer all information or require multiple files.
        self.logger_download.info("Checking and possibly downloading dataset with name %s" % str(self.dataset_name))

        # Default functions to load a dataset.
        if self.data_directory_name is not None:
            self.setup_dataset_main(self.data_main_dir, logger=self.logger_download)
            self.setup_dataset_dir(self.data_main_dir, self.data_directory_name, logger=self.logger_download)

        if self.download_url is not None:
            self.download_database(
                os.path.join(self.data_main_dir, self.data_directory_name), self.download_url,
                self.download_file_name, overwrite=self.download_reload, logger=self.logger_download)

        if self.unpack_tar:
            self.unpack_tar_file(
                os.path.join(self.data_main_dir, self.data_directory_name), self.download_file_name,
                self.unpack_directory_name, overwrite=self.download_reload, logger=self.logger_download)

        if self.unpack_zip:
            self.unpack_zip_file(
                os.path.join(self.data_main_dir, self.data_directory_name), self.download_file_name,
                self.unpack_directory_name, overwrite=self.download_reload, logger=self.logger_download)

        if self.extract_gz:
            self.extract_gz_file(
                os.path.join(self.data_main_dir, self.data_directory_name), self.download_file_name,
                self.extract_file_name, overwrite=self.download_reload, logger=self.logger_download)

    @staticmethod
    def setup_dataset_main(data_main_dir, logger=None):
        """Make the main-directory for all datasets to store data.

        Args:
           data_main_dir (str): Path to create directory.
           logger: Logger to print information or warnings.
        """
        os.makedirs(data_main_dir, exist_ok=True)
        if logger is not None:
            logger.info("Dataset directory located at %s" % data_main_dir)

    @staticmethod
    def setup_dataset_dir(data_main_dir: str, data_directory: str, logger=None):
        """Make directory for each dataset.

        Args:
            data_main_dir (str): Path-location of the directory for all datasets.
            data_directory (str): Path of the directory for specific dataset to create.
            logger: Logger to print information or warnings.
        """
        if not os.path.exists(os.path.join(data_main_dir, data_directory)):
            os.mkdir(os.path.join(data_main_dir, data_directory))
        else:
            if logger is not None:
                logger.info("Dataset directory found. Done.")

    @staticmethod
    def download_database(path: str, download_url: str, filename: str, overwrite: bool = False, logger=None):
        """Download dataset file.

        Args:
            path (str): Target filepath to store file (without filename).
            download_url (str): String of the download url to catch database from.
            filename (str): Name the dataset is downloaded to.
            overwrite (bool): Overwrite existing database. Default is False.
            logger: Logger to print information or warnings.

        Returns:
            os.path: Filepath of downloaded file.
        """
        def logg_info(msg):
            if logger is not None:
                logger.info(msg)

        if os.path.exists(os.path.join(path, filename)) is False or overwrite:
            logg_info("Downloading dataset... ")
            r = requests.get(download_url, allow_redirects=True)
            open(os.path.join(path, filename), 'wb').write(r.content)
        else:
            logg_info("Dataset found. Done.")
        return os.path.join(path, filename)

    @staticmethod
    def unpack_tar_file(path: str, filename: str, unpack_directory: str, overwrite: bool = False, logger=None):
        """Extract tar-file.

        Args:
            path (str): Filepath where the tar-file to extract is located.
            filename (str): Name of the dataset to extract.
            unpack_directory (str): Directory to extract data to.
            overwrite (bool): Overwrite existing database. Default is False.
            logger: Logger to print information or warnings.

        Returns:
            os.path: Filepath of the extracted dataset folder.
        """
        def logg_info(msg):
            if logger is not None:
                logger.info(msg)

        if not os.path.exists(os.path.join(path, unpack_directory)):
            logg_info("Creating directory... ")
            os.mkdir(os.path.join(path, unpack_directory))
        else:
            logg_info("Directory for extraction exists. Done.")
            if not overwrite:
                logg_info("Not extracting tar File. Stopped.")
                return os.path.join(path, unpack_directory)  # Stop extracting here

        logg_info("Read tar file... ")
        archive = tarfile.open(os.path.join(path, filename), "r")
        # Filelistnames = archive.getnames()

        logg_info("Extracting tar file... ")
        archive.extractall(os.path.join(path, unpack_directory))
        archive.close()
        return os.path.join(path, unpack_directory)

    @staticmethod
    def unpack_zip_file(path: str, filename: str, unpack_directory: str, overwrite: bool = False, logger=None):
        """Extract zip-file.

        Args:
            path (str): Filepath where the zip-file to extract is located.
            filename (str): Name of the dataset to extract.
            unpack_directory (str): Directory to extract data to.
            overwrite (bool): Overwrite existing database. Default is False.
            logger: Logger to print information or warnings.

        Returns:
            os.path: Filepath of the extracted dataset folder.
        """
        def logg_info(msg):
            if logger is not None:
                logger.info(msg)

        if os.path.exists(os.path.join(path, unpack_directory)):
            logg_info("Directory for extraction exists. Done.")
            if not overwrite:
                logg_info("Not extracting zip file. Stopped.")
                return os.path.join(path, unpack_directory)

        logg_info("Read zip file ... ")
        archive = zipfile.ZipFile(os.path.join(path, filename), "r")
        # Filelistnames = archive.getnames()
        logg_info("Extracting zip file...")
        archive.extractall(os.path.join(path, unpack_directory))
        archive.close()
        return os.path.join(path, unpack_directory)

    @staticmethod
    def extract_gz_file(path: str, filename: str, out_filename: str = None,
                        overwrite: bool = False, logger=None):
        """Extract gz-file.

        Args:
            path (str): Filepath where the gz-file to extract is located.
            filename (str): Name of the gz-file to extract.
            out_filename (str): Name of the extracted file.
            overwrite (bool): Overwrite existing database. Default is False.
            logger: Logger to print information or warnings.

        Returns:
            os.path: Filepath of the extracted file.
        """
        def logg_info(msg):
            if logger is not None:
                logger.info(msg)

        if out_filename is None:
            out_filename = filename.replace(".gz", "")

        if os.path.exists(os.path.join(path, out_filename)):
            logg_info("Extracted file exists. Done.")
            if not overwrite:
                logg_info("Not extracting gz-file. Stopped.")
                return os.path.join(path, out_filename)

        logg_info("Extract gz-file ... ")

        with gzip.open(os.path.join(path, filename), 'rb') as f_in:
            with open(os.path.join(path, out_filename), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        return os.path.join(path, out_filename)

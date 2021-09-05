import os
import numpy as np
import requests
import tarfile
import zipfile

from kgcnn.utils.adj import get_angle_indices, coordinates_to_distancematrix, invert_distance, \
    define_adjacency_from_distance, sort_edge_indices, get_angle


class MemoryGraphDataset:
    fits_in_memory = True
    dataset_name = None

    def __init__(self, **kwargs):
        self.length = None

        self.node_attributes = None
        self.node_labels = None
        self.node_degree = None
        self.node_symbol = None
        self.node_number = None

        self.edge_indices = None
        self.edge_attributes = None
        self.edge_labels = None
        self.edge_number = None

        self.graph_labels = None
        self.graph_attributes = None
        self.graph_number = None
        self.graph_size = None
        self.graph_adjacency = None  # Only for one-graph datasets like citation networks


class MemoryGeometricGraphDataset(MemoryGraphDataset):

    def __init__(self, **kwargs):
        super(MemoryGeometricGraphDataset, self).__init__(**kwargs)

        self.node_coordinates = None

        self.range_indices = None
        self.range_attributes = None
        self.range_labels = None

        self.angle_indices = None
        self.angle_labels = None
        self.angle_attributes = None

    def set_range(self, max_distance=4, max_neighbours=15, do_invert_distance=False, self_loops=False, exclusive=True):
        """Define range in euclidean space for interaction or edge-like connections. Requires node coordinates."""

        coord = self.node_coordinates
        if self.node_coordinates is None:
            print("WARNING:kgcnn: Coordinates are not set for `GeometricGraph`. Can not make graph.")
            return self

        edge_idx = []
        edges = []
        for i in range(len(coord)):
            xyz = coord[i]
            dist = coordinates_to_distancematrix(xyz)
            # cons = get_connectivity_from_inversedistancematrix(invdist,ats)
            cons, indices = define_adjacency_from_distance(dist, max_distance=max_distance,
                                                           max_neighbours=max_neighbours,
                                                           exclusive=exclusive, self_loops=self_loops)
            mask = np.array(cons, dtype=np.bool)
            dist_masked = dist[mask]

            if do_invert_distance:
                dist_masked = invert_distance(dist_masked)

            # Need at least one feature dimension
            if len(dist_masked.shape) <= 1:
                dist_masked = np.expand_dims(dist_masked, axis=-1)
            edges.append(dist_masked)
            edge_idx.append(indices)

        self.range_attributes = edges
        self.range_indices = edge_idx
        return self

    def set_angle(self):
        # We need to sort indices
        for i, x in enumerate(self.range_indices):
            order = np.arange(len(x))
            x_sorted, reorder = sort_edge_indices(x, order)
            self.range_indices[i] = x_sorted
            # Must sort attributes accordingly!
            if self.range_attributes is not None:
                self.range_attributes[i] = self.range_attributes[i][reorder]
            if self.range_labels is not None:
                self.range_labels[i] = self.range_labels[i][reorder]

        # Compute angles
        a_indices = []
        a_angle = []
        for i, x in enumerate(self.range_indices):
            temp = get_angle_indices(x)
            a_indices.append(temp[2])
            if self.node_coordinates is not None:
                a_angle.append(get_angle(self.node_coordinates[i], temp[1]))
        self.angle_indices = a_indices
        self.angle_attributes = a_angle
        return self


class DownloadDataset:
    """Base layer for datasets. Provides functions for download and unzip of the data.
    Dataset-specific functions like prepare_data() must be implemented in subclasses.
    Information about the dataset can be set with class properties.

    """
    dataset_name = None
    file_name = None
    data_main_dir = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets")
    data_directory = None
    unpack_directory = None
    download_url = None
    unpack_tar = False
    unpack_zip = False
    fits_in_memory = False
    require_prepare_data = False

    def __init__(self, reload=False, verbose=1, **kwargs):
        """Base initialization function for downloading and extracting the data.

        Args:
            reload (bool): Whether to reload the data and make new dataset. Default is False.
            verbose (int): Print progress or info for processing where 0=silent. Default is 1.
        """

        # Properties that could or should be set by read_in_memory() and get_graph() if memory is not an issue.
        # Some datasets do not offer all information.
        if verbose > 1:
            print("INFO:kgcnn: Checking and possibly downloading dataset with name %s" % str(self.dataset_name))

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

        if self.require_prepare_data:
            # Used if a standard processing of the data has to be done and save e.g. a pickled version for example.
            self.prepare_data(overwrite=reload, verbose=verbose)

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

    def prepare_data(self, overwrite=False, verbose=1, **kwargs):
        """Optional function for child classes to prepare the data, like compress etc."""
        pass

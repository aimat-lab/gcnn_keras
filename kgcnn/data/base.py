import os
import numpy as np
import requests
import tarfile
import zipfile
import gzip
import shutil

from kgcnn.utils.adj import get_angle_indices, coordinates_to_distancematrix, invert_distance, \
    define_adjacency_from_distance, sort_edge_indices, get_angle


class MemoryGraphDataset:

    fits_in_memory = True

    def __init__(self, **kwargs):
        self.length = None

        self.node_attributes = None
        self.node_labels = None
        self.node_degree = None
        self.node_symbol = None
        self.node_number = None

        self.edge_indices = None
        self.edge_indices_reverse_pairs = None
        self.edge_attributes = None
        self.edge_labels = None
        self.edge_number = None
        self.edge_symbol = None

        self.graph_labels = None
        self.graph_attributes = None
        self.graph_number = None
        self.graph_size = None
        self.graph_adjacency = None  # Only for one-graph datasets like citation networks

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

    def set_edge_indices_reverse_pairs(self):
        """Computes the index map of the reverse edge for each of the edges if available."""
        all_index_map = []
        # This must be done in mini-batches graphs are too large
        for edge_idx in self.edge_indices:
            if len(edge_idx) == 0:
                all_index_map.append(np.array([], dtype="int"))
                continue
            edge_idx_rev = np.flip(edge_idx, axis=-1)
            edge_pos, rev_pos = np.where(
                np.all(np.expand_dims(edge_idx, axis=1) == np.expand_dims(edge_idx_rev, axis=0), axis=-1))
            # May have duplicates
            ege_pos_uni, uni_pos = np.unique(edge_pos, return_index=True)
            rev_pos_uni = rev_pos[uni_pos]
            edge_map = np.empty(len(edge_idx), dtype="int")
            edge_map.fill(np.nan)
            edge_map[ege_pos_uni] = rev_pos_uni
            all_index_map.append(np.expand_dims(edge_map, axis=-1))

        self.edge_indices_reverse_pairs = all_index_map
        return self


class MemoryGeometricGraphDataset(MemoryGraphDataset):
    r"""Subclass of :obj:``MemoryGraphDataset``. It expands the graph dataset with range and angle properties.
    The range-attributes and range-indices are just like edge-indices but refer to a geometric annotation. This allows
    to have geometric range-connections and topological edges separately. The label 'range' is synonym for a geometric
    edge.

    """

    def __init__(self, **kwargs):
        super(MemoryGeometricGraphDataset, self).__init__(**kwargs)

        self.node_coordinates = None

        self.range_indices = None
        self.range_attributes = None
        self.range_labels = None

        self.angle_indices = None
        self.angle_labels = None
        self.angle_attributes = None

    def set_range_from_edges(self, do_invert_distance=False):
        """Simply assign the range connections identical to edges."""
        if self.edge_indices is None:
            raise ValueError("ERROR:kgcnn: Edge indices are not set. Can not infer range definition.")
        coord = self.node_coordinates

        if self.node_coordinates is None:
            print("WARNING:kgcnn: Coordinates are not set for `GeometricGraph`. Can not make graph.")
            return self

        self.range_indices = [np.array(x, dtype="int") for x in self.edge_indices]  # make a copy here

        edges = []
        for i in range(len(coord)):
            idx = self.range_indices[i]
            xyz = coord[i]
            dist = np.sqrt(np.sum(np.square(xyz[idx[:, 0]] - xyz[idx[:, 1]]), axis=-1, keepdims=True))
            if do_invert_distance:
                dist = invert_distance(dist)
            edges.append(dist)
        self.range_attributes = edges
        return self

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
    r"""Base layer for datasets. Provides functions for download and unzip of the data.
    Dataset-specific functions like prepare_data() must be implemented in subclasses.
    Note that ``DownloadDataset`` uses a main directory located at '~/.kgcnn/datasets' for downloading datasets.

    """

    data_main_dir = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets")

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
        self.dataset_name = dataset_name
        self.download_file_name = download_file_name
        self.data_directory_name = data_directory_name
        self.unpack_directory_name = unpack_directory_name
        self.extract_file_name = extract_file_name
        self.download_url = download_url
        self.unpack_tar = unpack_tar
        self.unpack_zip = unpack_zip
        self.extract_gz = extract_gz
        # self.reload = reload
        # self.verbose = 1

        # Properties that could or should be set by read_in_memory() and get_graph() if memory is not an issue.
        # Some datasets do not offer all information.
        if verbose > 1:
            print("INFO:kgcnn: Checking and possibly downloading dataset with name %s" % str(self.dataset_name))

        # Default functions to load a dataset.
        if self.data_directory_name is not None:
            self.setup_dataset_main(self.data_main_dir, verbose=verbose)
            self.setup_dataset_dir(self.data_main_dir, self.data_directory_name, verbose=verbose)

        if self.download_url is not None:
            self.download_database(os.path.join(self.data_main_dir, self.data_directory_name), self.download_url,
                                   self.download_file_name, overwrite=reload, verbose=verbose)

        if self.unpack_tar:
            self.unpack_tar_file(os.path.join(self.data_main_dir, self.data_directory_name), self.download_file_name,
                                 self.unpack_directory_name, overwrite=reload, verbose=verbose)

        if self.unpack_zip:
            self.unpack_zip_file(os.path.join(self.data_main_dir, self.data_directory_name), self.download_file_name,
                                 self.unpack_directory_name, overwrite=reload, verbose=verbose)

        if self.extract_gz:
            self.extract_gz_file(os.path.join(self.data_main_dir, self.data_directory_name), self.download_file_name,
                                 self.extract_file_name, overwrite=reload, verbose=verbose)

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

import numpy as np
import pandas as pd
import os

from kgcnn.utils.adj import get_angle_indices, coordinates_to_distancematrix, invert_distance, \
    define_adjacency_from_distance, sort_edge_indices, get_angle, add_edges_reverse_indices, \
    rescale_edge_weights_degree_sym, add_self_loops_to_edge_indices, compute_reverse_edges_index_map
from kgcnn.utils.data import save_pickle_file, load_pickle_file


class MemoryGraphList:
    r"""Class to store a list of graph properties in memory.
    The graph properties are defined by tensor-like numpy arrays for indices, attributes, labels, symbol etc. .
    They are added in form of a list as class attributes to the instance of this class.
    Graph related properties must have a special prefix to be noted as graph property.
    Prefix are `node_`, `edge_` and `graph_` for their node, edge and graph properties, respectively.

    .. code-block:: python

        from kgcnn.data.base import MemoryGraphList
        data = MemoryGraphList()
        data.edge_indices = [np.array([[0, 1], [1, 0]])]
        data.node_labels = [np.array([[0], [1]])]
        print(data.edge_indices, data.node_labels)

    For now, only the length of the different properties are checked to be the same when assigning the (new)
    attributes. Functions to modify graph properties are
    provided with this class, like for example :obj:`sort_edge_indices`, which expect list of numpy arrays. Please
    find functions in :class:`kgcnn.utils.adj` and their documentation for further details.
    """

    def __init__(self, length: int = None):
        r"""Initialize an empty :obj:`MemoryGraphList` instance. The expected length can be already provided to
        throw an error if a graph list does not match the length of the dataset. If you want to expand the list or
        namespace of accepted reserved graph prefix identifier, you can expand :obj:`_reserved_graph_property_prefix`.

        Args:
            length (int): Length of the graph list.
        """
        self._length = length
        self._reserved_graph_property_prefix = ["node_", "edge_", "graph_"]

        # Assign empty node attributes frequently used by subclasses.
        self.node_coordinates = None
        self.node_attributes = None
        self.node_labels = None
        self.node_degree = None
        self.node_symbol = None
        self.node_number = None

        # Assign empty edge attributes frequently used by subclasses.
        self.edge_indices = None
        self.edge_indices_reverse = None
        self.edge_attributes = None
        self.edge_labels = None
        self.edge_number = None
        self.edge_symbol = None
        self.edge_weights = None

        # Assign empty graph attributes frequently used by subclasses.
        self.graph_labels = None
        self.graph_attributes = None
        self.graph_number = None
        self.graph_size = None

    def _find_graph_properties(self, prop_prefix):
        return [x for x in list(self.__dict__.keys()) if prop_prefix == x[:len(prop_prefix)]]

    def _find_all_graph_properties(self):
        return [x for x in list(self.__dict__.keys()) for prop_prefix in
                self._reserved_graph_property_prefix if prop_prefix == x[:len(prop_prefix)]]

    def __setattr__(self, key, value):
        if hasattr(self, "_reserved_graph_property_prefix"):
            if any([x == key[:len(x)] for x in self._reserved_graph_property_prefix]) and value is not None:
                if self._length is None:
                    self._length = len(value)
                else:
                    if self._length != len(value):
                        raise ValueError("len() of %s does not match len() of graph list." % key)

        super(MemoryGraphList, self).__setattr__(key, value)

    @property
    def length(self):
        """Return the current length of this instance."""
        return self._length

    @length.setter
    def length(self, value: int):
        r"""Reset the nominal length of all graph attributes. Sets all graph properties to :obj:`None`,
        which do not match the new length.

        Args:
            value (int): Length of graph properties for this instance.
        """
        if value is None:
            self._length = None
            for x in self._find_all_graph_properties():
                setattr(self, x, None)
        else:
            for x in self._find_all_graph_properties():
                ax = getattr(self, x)
                if ax is not None:
                    if len(ax) != value:
                        setattr(self, x, None)
            self._length = int(value)

    def _operate_on_edges(self, operation, _prefix_attributes: str = "edge_", **kwargs):
        r"""Wrapper to run a certain function on all edge related properties. The indices attributes must be defined
        and must be composed of :obj:`_prefix_attributes` and 'indices'.

        Args:
              operation (callable): Function to apply to a list of all edge arrays.
                First entry is assured to be indices.
              _prefix_attributes (str): Prefix for attributes to identify as edges.
              kwargs: Kwargs for operation function call.
        """
        if getattr(self, _prefix_attributes + "indices") is None:
            raise ValueError("Can not operate on edges, as indices are not defined.")

        # Determine all linked edge attributes, that are not None.
        edge_linked = self._find_graph_properties(_prefix_attributes)
        # Edge indices is always at first position!
        edge_linked = [_prefix_attributes + "indices"] + [x for x in edge_linked if x != _prefix_attributes + "indices"]
        no_nan_edge_prop = [x for x in edge_linked if getattr(self, x) is not None]
        non_nan_edge = [getattr(self, x) for x in no_nan_edge_prop]

        # Create empty list for all edges.
        new_edges = [[] for _ in non_nan_edge]
        for eds in zip(*non_nan_edge):
            added_edge = operation(*eds, **kwargs)
            # If dataset only has edge indices, fun_operation is expected to only return array not list!
            # This restricts the type of fun_operation used with this method.
            if len(no_nan_edge_prop) == 1:
                added_edge = [added_edge]
            for i, x in enumerate(new_edges):
                x.append(added_edge[i])

        # Set all new edge attributes.
        for i, at in enumerate(no_nan_edge_prop):
            setattr(self, at, new_edges[i])

        return self

    def set_edge_indices_reverse(self):
        r"""Computes the index map of the reverse edge for each of the edges if available. This can be used by a model
        to directly select the corresponding edge of :math:`(j, i)` which is :math:`(i, j)`.
        Does not affect other edge-properties, only creates a map on edge indices. Edges that do not have a reverse
        pair get a `nan` as map index. If there are multiple edges, the first encounter is assigned.

        .. warning::
            Reverse maps are not recomputed if you use e.g. :obj:`sort_edge_indices` or redefine edges.

        Returns:
            self
        """
        self.edge_indices_reverse = [np.expand_dims(compute_reverse_edges_index_map(x), axis=-1) for x in
                                     self.edge_indices]
        return self

    def make_undirected_edges(self, remove_duplicates: bool = True, sort_indices: bool = True):
        r"""Add edges :math:`(j, i)` for :math:`(i, j)` if there is no edge :math:`(j, i)`.
        With :obj:`remove_duplicates` an edge can be added even though there is already and edge at :math:`(j, i)`.
        For other edge tensors, like the attributes or labels, the values of edge :math:`(i, j)` is added in place.
        Requires that :obj:`edge_indices` property is assigned.

        Args:
            remove_duplicates (bool): Whether to remove duplicates within the new edges. Default is True.
            sort_indices (bool): Sort indices after adding edges. Default is True.

        Returns:
            self
        """
        self._operate_on_edges(add_edges_reverse_indices, remove_duplicates=remove_duplicates,
                               sort_indices=sort_indices)
        return self

    def add_edge_self_loops(self, remove_duplicates: bool = True, sort_indices: bool = True, fill_value: int = 0):
        r"""Add self loops to the each graph property. The function expects the property :obj:`edge_indices`
        to be defined. By default the edges are also sorted after adding the self-loops.
        All other edge properties are filled with :obj:`fill_value`.

        Args:
            remove_duplicates (bool): Whether to remove duplicates. Default is True.
            sort_indices (bool): To sort indices after adding self-loops. Default is True.
            fill_value (in): The fill_value for all other edge properties.

        Returns:
            self
        """
        self._operate_on_edges(add_self_loops_to_edge_indices, remove_duplicates=remove_duplicates,
                               sort_indices=sort_indices, fill_value=fill_value)
        return self

    def sort_edge_indices(self):
        r"""Sort edge indices and all edge-related properties. The index list is sorted for the first entry.

        Returns:
            self
        """
        self._operate_on_edges(sort_edge_indices)
        return self

    def normalize_edge_weights_sym(self):
        r"""Normalize :obj:`edge_weights` using the node degree of each row or column of the adjacency matrix.
        Normalize edge weights as :math:`\tilde{e}_{i,j} = d_{i,i}^{-0.5} \, e_{i,j} \, d_{j,j}^{-0.5}`.
        The node degree is defined as :math:`D_{i,i} = \sum_{j} A_{i, j}`. Requires the property :obj:`edge_indices`.
        Does not affect other edge-properties and only sets :obj:`edge_weights`.

        Returns:
            self
        """
        if self.edge_indices is None:
            raise ValueError("Can scale adjacency matrix, as graph indices are not defined.")
        if self.edge_weights is None:
            self.edge_weights = [np.ones_like(x[:, :1]) for x in self.edge_indices]

        new_weights = []
        for idx, edw in zip(self.edge_indices, self.edge_weights):
            new_weights.append(rescale_edge_weights_degree_sym(idx, edw))

        self.edge_weights = new_weights
        return self


class MemoryGeometricGraphList(MemoryGraphList):
    r"""Subclass of :obj:`MemoryGraphList`. It expands the graph list with range and angle properties.
    The range-attributes and range-indices are just like edge-indices but refer to a geometric annotation. This allows
    to have geometric range-connections and topological edges separately. The label 'range' is synonym for a geometric
    edge. They are characterized by the prefix `range_` and `angle_` and are also
    checked for length when assigning attributes to the instances of this class.

    .. code-block:: python

        from kgcnn.data.base import MemoryGeometricGraphList
        dataset = MemoryGeometricGraphList(length=1)
        dataset.node_coordinates = [np.array([[1, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]])]
        print(dataset.node_coordinates)
        dataset.set_range(max_distance=1.5, max_neighbours=10, self_loops=False)
        print(dataset.range_indices, dataset.range_attributes)

    Having additional geometric edges is mainly required for structures like molecules that have explicit bonds
    (chemical structure) but also coordinates and long range interactions that depend on their mutual distance
    beyond directly bonded atom pairs.
    """

    def __init__(self, **kwargs):
        r"""Initialize a :obj:`MemoryGeometricGraphDataset` instance. See :obj:`MemoryGraphDataset` for definition
        of arguments. Already sets empty `range` and `angle` attributes.

        Args:
            kwargs: Arguments that are passed to :obj:`MemoryGraphDataset` base class.
        """
        super(MemoryGeometricGraphList, self).__init__(**kwargs)
        self._reserved_graph_property_prefix = self._reserved_graph_property_prefix + ["range_", "angle_"]

        # Extend nodes to have coordinates.
        self.node_coordinates = None

        self.range_indices = None
        self.range_attributes = None
        self.range_labels = None

        self.angle_indices = None
        self.angle_indices_nodes = None
        self.angle_labels = None
        self.angle_attributes = None

    def set_range_from_edges(self, do_invert_distance: bool = False):
        r"""Assigns range indices and attributes (distance) from the definition of edge indices. This operations
        requires the attributes :obj:`node_coordinates` and :obj:`edge_indices` to be set. That also means that
        :obj:`range_indices` will be equal to :obj:`edge_indices`.

        Args:
            do_invert_distance (bool): Invert distance when computing  :obj:`range_attributes`. Default is False.

        Returns:
            self
        """
        if self.edge_indices is None:
            raise ValueError("Edge indices are not set. Can not infer range definition.")
        coord = self.node_coordinates

        # We make a copy here of the edge indices.
        self.range_indices = [np.array(x, dtype="int") for x in self.edge_indices]

        if self.node_coordinates is None:
            print("Coordinates are not set for `GeometricGraph`. Can not make graph.")
            return self

        edges = []
        for i in range(len(coord)):
            idx = self.range_indices[i]
            # Assuming all list are numpy arrays.
            xyz = coord[i]
            dist = np.sqrt(np.sum(np.square(xyz[idx[:, 0]] - xyz[idx[:, 1]]), axis=-1, keepdims=True))
            if do_invert_distance:
                dist = invert_distance(dist)
            edges.append(dist)
        self.range_attributes = edges
        return self

    def set_range(self, max_distance: float = 4.0, max_neighbours: int = 15,
                  do_invert_distance: bool = False, self_loops: bool = False, exclusive: bool = True):
        r"""Define range in euclidean space for interaction or edge-like connections. The number of connection is
        determines based on a cutoff radius and a maximum number of neighbours or both.
        Requires :obj:`node_coordinates` and :obj:`edge_indices` to be set.
        The distance is stored in :obj:`range_attributes`.

        Args:
            max_distance (float): Maximum distance or cutoff radius for connections. Default is 4.0.
            max_neighbours (int): Maximum number of allowed neighbours for a node. Default is 15.
            do_invert_distance (bool): Whether to invert the the distance. Default is False.
            self_loops (bool): If also self-interactions with distance 0 should be considered. Default is False.
            exclusive (bool): Whether both max_neighbours and max_distance must be fulfilled. Default is True.

        Returns:
            self
        """

        coord = self.node_coordinates
        if self.node_coordinates is None:
            print("Coordinates are not set for `GeometricGraph`. Can not make graph.")
            return self

        edge_idx = []
        edges = []
        for i in range(len(coord)):
            xyz = coord[i]
            # Compute distance matrix here. May be problematic for too large graphs.
            dist = coordinates_to_distancematrix(xyz)
            cons, indices = define_adjacency_from_distance(dist, max_distance=max_distance,
                                                           max_neighbours=max_neighbours,
                                                           exclusive=exclusive, self_loops=self_loops)
            mask = np.array(cons, dtype="bool")
            dist_masked = dist[mask]

            if do_invert_distance:
                dist_masked = invert_distance(dist_masked)

            # Need one feature dimension.
            if len(dist_masked.shape) <= 1:
                dist_masked = np.expand_dims(dist_masked, axis=-1)
            edges.append(dist_masked)
            edge_idx.append(indices)

        # Assign attributes to instance.
        self.range_attributes = edges
        self.range_indices = edge_idx
        return self

    def set_angle(self, prefix_indices: str = "range", allow_multi_edges: bool = False, compute_angles: bool = True):
        r"""Find possible angles between geometric range connections defined by :obj:`range_indices`.
        Which edges form angles is stored in :obj:`angle_indices`.
        One can also change :obj:`prefix_indices` to `edge` to compute angles between edges instead
        of range connections.

        .. warning::
            Angles are not recomputed if you use :obj:`set_range` or redefine edges.

        Args:
            prefix_indices (str): Prefix for edge-like attributes to pick indices from. Default is `range`.
            allow_multi_edges (bool): Whether to allow angles between 'i<-j<-i', which gives 0 degree angle, if they
                the nodes are unique. Default is False.
            compute_angles (bool): Whether to also compute angles

        Returns:
            self
        """

        # Compute angles
        e_indices = getattr(self, prefix_indices + "_indices")
        a_indices = []
        a_triples = []
        for i, x in enumerate(e_indices):
            temp = get_angle_indices(x, allow_multi_edges=allow_multi_edges)
            a_indices.append(temp[2])
            a_triples.append(temp[1])
        self.angle_indices = a_indices
        self.angle_indices_nodes = a_triples
        # Also compute angles
        if compute_angles:
            a_angle = []
            for i, x in enumerate(a_triples):
                a_angle.append(get_angle(self.node_coordinates[i], x))
            self.angle_attributes = a_angle
        return self


class MemoryGraphDataset(MemoryGeometricGraphList):
    r"""Dataset class for lists of graph tensor properties that can be cast into the :obj:`tf.RaggedTensor` class.
    The graph list is expected to only store numpy arrays in place of the each node or edge information!

    .. note::
        Each graph attribute is expected to be a python list or iterable object containing numpy arrays.
        For example, the special attribute of :obj:`edge_indices` is expected to be a list of arrays of
        shape `(Num_edges, 2)` with the indices of node connections.
        The node attributes in :obj:`node_attributes` are numpy arrays of shape `(Num_nodes, Num_features)`.

    The Memory Dataset class inherits from :obj:`MemoryGeometricGraphList` and has further information
    about a location on disk, i.e. a file directory and a file name as well as a name of the dataset.

    .. code-block:: python

        from kgcnn.data.base import MemoryGraphDataset
        dataset = MemoryGraphDataset(data_directory="", dataset_name="Example", length=1)
        dataset.edge_indices = [np.array([[1, 0], [0, 1]])]
        dataset.edge_labels = [np.array([[0], [1]])]
        print(dataset.edge_indices, dataset.edge_labels)
        dataset.sort_edge_indices()
        print(dataset.edge_indices, dataset.edge_labels)

    The file directory and file name are not used directly. However, for :obj:`load()` and :obj:`safe()`,
    the default is constructed from the data directory and dataset name. File name and file directory is reserved for
    child classes.
    """

    fits_in_memory = True

    def __init__(self,
                 data_directory: str = None,
                 dataset_name: str = None,
                 file_name: str = None,
                 file_directory: str = None,
                 length: int = None,
                 verbose: int = 1, **kwargs):
        r"""Initialize a base class of :obj:`MemoryGraphDataset`.

        Args:
            data_directory (str): Full path to directory of the dataset. Default is None.
            file_name (str): Generic filename for dataset to read into memory like a 'csv' file. Default is None.
            file_directory (str): Name or relative path from :obj:`data_directory` to a directory containing sorted
                files. Default is None.
            dataset_name (str): Name of the dataset. Important for naming and saving files. Default is None.
            length (int): Length of the dataset, if known beforehand. Default is None.
            verbose (int): Print progress or info for processing, where 0 is silent. Default is 1.
        """
        super(MemoryGraphDataset, self).__init__(length=length)
        # For logging.
        self.verbose = verbose
        # Dataset information on file. Take care of path and name specifics here.
        self.data_directory = data_directory
        self.file_name = file_name
        self.file_directory = file_directory
        self.dataset_name = dataset_name

        # Data Frame for information.
        self.data_frame = None

        # Check if no wrong kwargs passed to init. Reserve for future compatibility.
        if len(kwargs) > 0:
            self.warning("Unknown kwargs for `MemoryGraphDataset`: {}".format(list(kwargs.keys())))

    @property
    def file_path(self):
        if self.data_directory is None:
            self.warning("Data directory is not set.")
            return None
        if not os.path.exists(self.data_directory):
            self.error("Data directory does not exist.")
        if self.file_name is None:
            self.warning("Can not determine file path.")
            return None
        return os.path.join(self.data_directory, self.file_name)

    @property
    def file_directory_path(self):
        if self.data_directory is None:
            self.warning("Data directory is not set.")
            return None
        if not os.path.exists(self.data_directory):
            self.error("Data directory does not exist.")
        if self.file_directory is None:
            self.warning("Can not determine file directory.")
            return None
        return os.path.join(self.data_directory, self.file_directory)

    def info(self, *args, **kwargs):
        """Logging information."""
        # Could use logger in the future.
        print_kwargs = {key: value for key, value in kwargs.items() if key not in ["verbose"]}
        verbosity_level = kwargs["verbose"] if "verbose" in kwargs else 0
        if self.verbose > verbosity_level:
            print("INFO:kgcnn:", *args, **print_kwargs)

    def warning(self, *args, **kwargs):
        """Logging information."""
        # Could use logger in the future.
        print_kwargs = {key: value for key, value in kwargs.items() if key not in ["verbose"]}
        verbosity_level = kwargs["verbose"] if "verbose" in kwargs else 0
        if self.verbose > verbosity_level:
            print("WARNING:kgcnn:", *args, **print_kwargs)

    def error(self, *args, **kwargs):
        """Logging information."""
        # Could use logger in the future.
        print_kwargs = {key: value for key, value in kwargs.items() if key not in ["verbose"]}
        verbosity_level = kwargs["verbose"] if "verbose" in kwargs else 0
        if self.verbose > verbosity_level:
            print("ERROR:kgcnn:", *args, **print_kwargs)

    def save(self, filepath: str = None):
        r"""Save all graph properties to as dictionary as pickled file. By default saves a file named
        :obj:`dataset_name.kgcnn.pickle` in :obj:`data_directory`.

        Args:
            filepath (str): Full path of output file. Default is None.
        """
        if filepath is None:
            filepath = os.path.join(self.data_directory, self.dataset_name + ".kgcnn.pickle")
        self.info("Pickle dataset...")
        save_pickle_file({x: getattr(self, x) for x in self._find_all_graph_properties()}, filepath)
        self.info("done")
        return self

    def load(self, filepath: str = None):
        r"""Load graph properties from a pickled file. By default loads a file named
        :obj:`dataset_name.kgcnn.pickle` in :obj:`data_directory`.

        Args:
            filepath (str): Full path of input file.
        """
        if filepath is None:
            filepath = os.path.join(self.data_directory, self.dataset_name + ".kgcnn.pickle")
        self.info("Load pickled dataset...")
        in_dict = load_pickle_file(filepath)
        for key, value in in_dict.items():
            if any([x == key[:len(x)] for x in self._reserved_graph_property_prefix]):
                setattr(self, key, value)
            else:
                self.warning("Not accepted graph property, ignore name %s" % key)
        self.info("done")
        return self

    def read_in_table_file(self, file_path: str = None, **kwargs):
        r"""Read a data frame in :obj:`data_frame` from file path. By default uses :obj:`file_name` and pandas.
        Checks for a '.csv' file and then for excel file endings. Meaning the file extension of file_path is ignored
        but must be any of the following '.csv', '.xls', '.xlsx', ... .

        Args:
            file_path (str): File path to table file. Default is None.
            kwargs: Kwargs for pandas :obj:`read_csv` function.

        Returns:
            self
        """
        if file_path is None:
            file_path = os.path.join(self.data_directory, self.file_name)
        file_path_base = os.path.splitext(file_path)[0]
        # file_extension_given = os.path.splitext(file_path)[1]

        for file_extension in [".csv"]:
            if os.path.exists(file_path_base + file_extension):
                self.data_frame = pd.read_csv(file_path_base + file_extension, **kwargs)
                return self
        for file_extension in [".xls", ".xlsx", ".xlsm", ".xlsb", ".odf", ".ods", ".odt"]:
            if os.path.exists(file_path_base + file_extension):
                self.data_frame = pd.read_excel(file_path_base + file_extension, **kwargs)
                return self

        self.warning("Unsupported data extension of %s for csv file." % file_path)
        return self

    def process_methods(self, hyper_data: dict):
        """Process methods for this dataset. That includes to set or execute methods of this class.

        Args:
            hyper_data (dict): Process hyper parameter for this dataset.

        Returns:
            self
        """
        # The order here is important. So for the moment we explicitly check for methods in hyper.
        methods_supported = ["set_range", "set_angle", "set_edge_indices_reverse", "normalize_edge_weights_sym"]
        if "set_range" in hyper_data:
            self.set_range(**hyper_data["set_range"])
        if "set_angle" in hyper_data:
            self.set_angle(**hyper_data['set_angle'])
        if "set_edge_indices_reverse" in hyper_data:
            self.set_edge_indices_reverse()
        if "normalize_edge_weights_sym" in hyper_data:
            self.normalize_edge_weights_sym()

        for key, value in hyper_data.items():
            if key not in methods_supported:
                self.warning("Can not process the method: %s" % key)
        return self

    def assert_valid_attribute(self, property_list: list):
        """Check whether dataset has requested properties.

        Args:
            property_list (list): List of properties that need to be available to a model for training.

        Returns:
            self
        """
        for x in property_list:
            if isinstance(x, dict):
                prop = x["name"]
            else:
                prop = x
            assert hasattr(self, prop), "Dataset does not have property on %s" % prop
            assert getattr(self, prop) is not None, "Empty property of %s is `None` or not set." % prop
        return self


MemoryGeometricGraphDataset = MemoryGraphDataset

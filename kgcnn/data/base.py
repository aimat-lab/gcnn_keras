import numpy as np
import os

from kgcnn.utils.adj import get_angle_indices, coordinates_to_distancematrix, invert_distance, \
    define_adjacency_from_distance, sort_edge_indices, get_angle, add_edges_reverse_indices, \
    rescale_edge_weights_degree_sym, add_self_loops_to_edge_indices
from kgcnn.utils.data import save_pickle_file, load_pickle_file


class MemoryGraphList:
    r"""Class to store a list of graph properties in memory.
    The graph properties are defined by tensor-like (numpy) arrays for indices, attributes, labels, symbol etc. .
    They are added in form of a list as class attributes to the instance of this class.
    Graph related properties must have a special prefix to be noted as graph property.
    Prefix are `node_`, `edge_` and `graph_` for their node, edge and graph properties, respectively.

    .. code-block:: python

        from kgcnn.data.base import MemoryGraphList
        data = MemoryGraphList()
        data.edge_indices = [np.array([[0, 1], [1, 0]])]
        data.node_labels = [np.array([[0], [1]])]
        print(data.edge_indices, data.node_labels)

    """

    def __init__(self, length: int = None):
        r"""Initialize an empty :obj:`MemoryGraphList` instance. The expected length can be already provided to
        throw an error if a graph list does not match the length of the dataset.

        Args:
            length (int): Length of the graph list.
        """
        self._length = length
        self._reserved_graph_property_prefix = ["node_", "edge_", "graph_"]

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
                        raise ValueError("ERROR:kgcnn: len() of %s does not match len() of graph list." % key)

        super(MemoryGraphList, self).__setattr__(key, value)

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        # Set all graph properties to none, which not match new length
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

    def save(self, filepath: str):
        """Save all graph properties to pickled file.

        Args:
            filepath (str): Full path of output file.
        """
        save_pickle_file({x: getattr(self, x) for x in self._find_all_graph_properties()}, filepath)
        return self

    def load(self, filepath: str):
        """Load graph properties from pickled file.

        Args:
            filepath (str): Full path of input file.
        """
        in_dict = load_pickle_file(filepath)
        for key, value in in_dict.items():
            if any([x == key[:len(x)] for x in self._reserved_graph_property_prefix]) and value is not None:
                setattr(self, key, value)
        return self


class MemoryGraphDataset(MemoryGraphList):
    r"""Dataset class for lists of graph tensor properties that can be cast into the tf.RaggedTensor class.
    The graph list is expected to only store numpy arrays in place of the each node or edge information.
    The Memory Dataset class inherits from :obj:`MemoryGraphList` had has further information about a location on file,
    i.e. a file directory and a file name as well as a name of the dataset. Functions to modify graph properties are
    provided with this class, like for example :obj:`sort_edge_indices`.

    .. code-block:: python

        from kgcnn.data.base import MemoryGraphDataset
        dataset = MemoryGraphDataset(data_directory="", dataset_name="Example", length=1)
        dataset.edge_indices = [np.array([[1, 0], [0, 1]])]
        dataset.edge_labels = [np.array([[0], [1]])]
        print(dataset.edge_indices, dataset.edge_labels)
        dataset.sort_edge_indices()
        print(dataset.edge_indices, dataset.edge_labels)

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
        self.length = length
        # Dataset information, if available.
        self.verbose = verbose
        self.data_directory = data_directory
        self.file_name = file_name
        self.file_directory = file_directory
        self.dataset_name = dataset_name
        if self.data_directory is None and isinstance(self.file_name, str):
            # Get path information from filename.
            self.data_directory = os.path.dirname(os.path.realpath(self.file_name))
        # Check if no wrong kwargs passed to init.
        if len(kwargs) > 0:
            print("WARNING:kgcnn: Unknown kwargs for `MemoryGraphDataset` found: {}".format(list(kwargs.keys())))

        self.node_attributes = None
        self.node_labels = None
        self.node_degree = None
        self.node_symbol = None
        self.node_number = None

        self.edge_indices = None
        self.edge_indices_reverse = None
        self.edge_attributes = None
        self.edge_labels = None
        self.edge_number = None
        self.edge_symbol = None
        self.edge_weights = None

        self.graph_labels = None
        self.graph_attributes = None
        self.graph_number = None
        self.graph_size = None

    def _log(self, *args, **kwargs):
        """Logging information."""
        # Could use logger in the future.
        print_kwargs = {key: value for key, value in kwargs.items() if key not in ["verbose"]}
        verbosity_level = kwargs["verbose"] if "verbose" in kwargs else 0
        if self.verbose > verbosity_level:
            print(*args, **print_kwargs)

    def _operate_on_edges(self, operation, **kwargs):
        r"""Wrapper to run a certain function on all edge related properties.

        Args:
              operation (callable): Function to apply to a list of edge arrays. First entry is assured indices.
              kwargs: Kwargs for operation function call.
        """
        if self.edge_indices is None:
            raise ValueError("ERROR:kgcnn: Can not make undirected edges, as edge indices are not defined.")

        # Determine all linked edge attributes, that are not None
        # Edge indices is always at first position
        edge_linked = self._find_graph_properties("edge_")
        edge_linked = ["edge_indices"] + [x for x in edge_linked if x != "edge_indices"]
        no_nan_edge_prop = [x for x in edge_linked if getattr(self, x) is not None]
        non_nan_edge = [getattr(self, x) for x in no_nan_edge_prop]

        # If no edge properties
        new_edges = [[] for _ in non_nan_edge]
        for eds in zip(*non_nan_edge):
            added_edge = operation(*eds, **kwargs)
            # If dataset only has edge indices, fun_operation is expected to only return array not list!
            if len(no_nan_edge_prop) == 1:
                added_edge = [added_edge]
            for i, x in enumerate(new_edges):
                x.append(added_edge[i])

        # Set new edges
        for i, at in enumerate(no_nan_edge_prop):
            setattr(self, at, new_edges[i])

        # Recompute the edge reverse pair map
        if self.edge_indices_reverse is not None:
            self.set_edge_indices_reverse()
        return self

    def set_edge_indices_reverse(self):
        r"""Computes the index map of the reverse edge for each of the edges if available. This can be used by a model
        to directly select the corresponding edge of :math:`(j, i)` which is :math:`(i, j)`.
        Does not affect other edge-properties.

        Returns:
            self
        """
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

        self.edge_indices_reverse = all_index_map
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

    def add_self_loops(self, remove_duplicates: bool = True, sort_indices: bool = True, fill_value: int = 0):
        r"""Add self loops to the each graph property. The function expects to have the property :obj:`edge_indices`.
        By default the edges are also sorted after adding the self-loops. All other edge properties are filled with
        :obj:`fill_value`.

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
        r"""Sort edge indices and all edge-related properties.

        Returns:
            self
        """
        self._operate_on_edges(sort_edge_indices)
        return self

    def normalize_edge_weights_sym(self):
        r"""Normalize :obj:`edge_weights` using the node degree of each row or column of the adjacency matrix.
        Normalize edge weights as :math:`\tilde(e)_{i,j} = d_{i,i}^{-0.5} e_{i,j} d_{j,j}^{-0.5}`.
        The node degree is defined as :math:`D_{i,i} = \sum_{j} A_{i, j}`. Requires property :obj:`edge_indices`.
        Does not affect other edge-properties.

        Returns:
            self
        """
        if self.edge_indices is None:
            raise ValueError("ERROR:kgcnn: Can scale adjacency matrix, as graph indices are not defined.")
        if self.edge_weights is None:
            self.edge_weights = [np.ones_like(x[:, :1]) for x in self.edge_indices]

        new_weights = []
        for idx, edw in zip(self.edge_indices, self.edge_weights):
            new_weights.append(rescale_edge_weights_degree_sym(idx, edw))

        self.edge_weights = new_weights
        return self

    def save(self, filepath: str = None):
        """Save all graph properties to pickled file.

        Args:
            filepath (str): Full path of output file.
        """
        if filepath is None:
            filepath = os.path.join(self.data_directory, self.dataset_name + ".kgcnn.pickle")
        self._log("INFO:kgcnn: Pickle dataset...")
        super(MemoryGraphDataset, self).save(filepath)
        return self

    def load(self, filepath: str = None):
        """Load graph properties from pickled file.

        Args:
            filepath (str): Full path of input file.
        """
        if filepath is None:
            filepath = os.path.join(self.data_directory, self.dataset_name + ".kgcnn.pickle")
        self._log("INFO:kgcnn: Load pickled dataset...")
        super(MemoryGraphDataset, self).load(filepath)
        return self


class MemoryGeometricGraphDataset(MemoryGraphDataset):
    r"""Subclass of :obj:``MemoryGraphDataset``. It expands the graph dataset with range and angle properties.
    The range-attributes and range-indices are just like edge-indices but refer to a geometric annotation. This allows
    to have geometric range-connections and topological edges separately. The label 'range' is synonym for a geometric
    edge.

    """

    def __init__(self, **kwargs):
        super(MemoryGeometricGraphDataset, self).__init__(**kwargs)
        self._reserved_graph_property_prefix = self._reserved_graph_property_prefix + ["range_", "angle_"]

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
            mask = np.array(cons, dtype="bool")
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

import numpy as np
import os

from kgcnn.utils.adj import get_angle_indices, coordinates_to_distancematrix, invert_distance, \
    define_adjacency_from_distance, sort_edge_indices, get_angle, add_edges_reverse_indices, \
    precompute_adjacency_scaled, make_adjacency_from_edge_indices, convert_scaled_adjacency_to_list, \
    add_self_loops_to_edge_indices


class MemoryGraphDataset:
    """Dataset class for storing lists of graph tensor properties that can be cast into the tf.RaggedTensor class.

    """

    fits_in_memory = True

    def __init__(self, data_directory: str = None, dataset_name: str = None, file_name: str = None, length: int = None,
                 verbose: int = 1, **kwargs):
        r"""Initialize a base class of :obj:`MemoryGraphDataset`.

        Args:
            file_name (str): Generic filename for dataset to read into memory.
                For some datasets this may not be required or a list of files must be provided. Default is None.
            data_directory (str): Full path to directory containing all dataset related files. Default is None.
            dataset_name (str): Name of the dataset. Important for naming and saving files. Default is None.
            length (int): Length of the dataset, if known beforehand. Default is None.
            verbose (int): Print progress or info for processing, where 0 is silent. Default is 1.
        """

        # Dataset information, if available.
        self.verbose = verbose
        self.data_directory = data_directory
        self.file_name = file_name
        if self.data_directory is None and isinstance(self.file_name, str):
            # Get path information from filename.
            self.data_directory = os.path.dirname(os.path.realpath(self.file_name))
        self.dataset_name = dataset_name
        self.length = length
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

    def set_edge_indices_reverse(self):
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

        self.edge_indices_reverse = all_index_map
        return self

    def _operate_on_edges(self, operation, **kwargs):
        """Wrapper to run a certain function on all edge related arrays.

        Args:
              operation (callable): Function to apply to a list of edge arrays.
              kwargs: Kwargs for fun_operation
        """
        if self.edge_indices is None:
            raise ValueError("ERROR:kgcnn: Can not make undirected edges, as graph indices are not defined.")

        # Determine all linked edge attributes, that are not None
        # Edge indices is always at first position
        _edge_linked = ["edge_indices", "edge_attributes", "edge_labels", "edge_number", "edge_symbol",
                        "edge_weights"]
        no_nan_edge_prop = [x for x in _edge_linked if getattr(self, x) is not None]
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

    def make_undirected_edges(self, remove_duplicates: bool = True, sort_indices: bool = True):
        r"""Add edges :math:`(j, i)` for :math:`(i, j)` if there is no edge :math:`(j, i)`. With `remove_duplicates`
        an edge can be added even though there is already and edge at :math:`(j, i)`.

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

        self._operate_on_edges(add_self_loops_to_edge_indices, remove_duplicates=remove_duplicates,
                               sort_indices=sort_indices, fill_value=fill_value)

    def normalize_edge_weights_sym(self, add_identity=True):

        if self.edge_indices is None:
            raise ValueError("ERROR:kgcnn: Can scale adjacency matrix, as graph indices are not defined.")
        if self.edge_weights is None:
            self.edge_weights = [np.ones_like(x[:, 0]) for x in self.edge_indices]
        self.edge_weights = [np.squeeze(x) if len(x.shape) > 1 else x for x in self.edge_weights]

        if add_identity:
            self.add_self_loops()

        new_weights = []
        for idx, edw in zip(self.edge_indices, self.edge_weights):
            # We cast to a sparse adjacency matrix using weights,
            adj = make_adjacency_from_edge_indices(idx, edw)
            adj = precompute_adjacency_scaled(adj, add_identity=add_identity)
            edi, ed = convert_scaled_adjacency_to_list(adj)
            new_weights.append(ed)
            # indices must not change
            assert len(edi) == len(idx), "ERROR:kgcnn: Edge indices changed when scaling weights."

        self.edge_weights = [np.expand_dims(x, axis=-1) if len(x.shape) <= 1 else x for x in self.edge_weights]
        return self

    def _print_info(self):
        pass

    def __str__(self):
        return self._print_info()


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

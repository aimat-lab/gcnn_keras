import numpy as np

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



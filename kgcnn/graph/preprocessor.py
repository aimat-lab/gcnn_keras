import numpy as np
import logging
from typing import Union
from kgcnn.graph.base import GraphPreProcessorBase, GraphDict
from kgcnn.graph.adj import get_angle_indices, coordinates_to_distancematrix, invert_distance, \
    define_adjacency_from_distance, sort_edge_indices, get_angle, add_edges_reverse_indices, \
    rescale_edge_weights_degree_sym, add_self_loops_to_edge_indices, compute_reverse_edges_index_map
from kgcnn.graph.geom import range_neighbour_lattice

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


class MakeUndirectedEdges(GraphPreProcessorBase):
    r"""Add edges :math:`(j, i)` for :math:`(i, j)` if there is no edge :math:`(j, i)`.
    With :obj:`remove_duplicates` an edge can be added even though there is already and edge at :math:`(j, i)`.
    For other edge tensors, like the attributes or labels, the values of edge :math:`(i, j)` is added in place.
    Requires that :obj:`edge_indices` property is assigned.

    Args:
        edge_indices (str): Name of indices in dictionary. Default is "edge_indices".
        edge_attributes (str): Name of related edge values or attributes.
            This can be a match-string or list of names. Default is "^edge_.*".
        remove_duplicates (bool): Whether to remove duplicates within the new edges. Default is True.
        sort_indices (bool): Sort indices after adding edges. Default is True.
    """
    def __init__(self, edge_indices: str = "edge_indices", edge_attributes: str = "^edge_.*",
                 remove_duplicates: bool = True, sort_indices: bool = True, name="make_undirected_edges", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"edge_indices": edge_indices, "edge_attributes": edge_attributes})
        self._to_assign = [edge_indices, edge_attributes]
        self._search = [edge_attributes]
        self._call_kwargs = {
            "remove_duplicates": remove_duplicates, "sort_indices": sort_indices}
        self._config_kwargs.update({"edge_indices": edge_indices, "edge_attributes": edge_attributes,
                                    **self._call_kwargs})

    def call(self, edge_indices: np.ndarray, edge_attributes: list, remove_duplicates: bool, sort_indices: bool):
        return add_edges_reverse_indices(
            edge_indices=edge_indices, *edge_attributes, remove_duplicates=remove_duplicates, sort_indices=sort_indices,
            return_nested=True)


class AddEdgeSelfLoops(GraphPreProcessorBase):
    r"""Add self loops to each graph property. The function expects the property :obj:`edge_indices`
    to be defined. By default, the edges are also sorted after adding the self-loops.
    All other edge properties are filled with :obj:`fill_value`.

    Args:
        edge_indices (str): Name of indices in dictionary. Default is "edge_indices".
        edge_attributes (str): Name of related edge values or attributes.
            This can be a match-string or list of names. Default is "^edge_.*".
        remove_duplicates (bool): Whether to remove duplicates. Default is True.
        sort_indices (bool): To sort indices after adding self-loops. Default is True.
        fill_value (in): The fill_value for all other edge properties.
    """
    def __init__(self, edge_indices: str = "edge_indices", edge_attributes: str = "^edge_.*",
                 remove_duplicates: bool = True, sort_indices: bool = True, fill_value: int = 0,
                 name="add_edge_self_loops", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"edge_indices": edge_indices, "edge_attributes": edge_attributes})
        self._to_assign = [edge_indices, edge_attributes]
        self._search = [edge_attributes]
        self._call_kwargs = {
            "remove_duplicates": remove_duplicates, "sort_indices": sort_indices, "fill_value": fill_value}
        self._config_kwargs.update({"edge_indices": edge_indices, "edge_attributes": edge_attributes,
                                    **self._call_kwargs})

    def call(self, *, edge_indices: np.ndarray, edge_attributes: list, remove_duplicates: bool, sort_indices: bool,
             fill_value: bool):
        return add_self_loops_to_edge_indices(
            edge_indices=edge_indices, *edge_attributes, remove_duplicates=remove_duplicates,
            sort_indices=sort_indices, fill_value=fill_value, return_nested=True)


class SortEdgeIndices(GraphPreProcessorBase):
    r"""Sort edge indices and all edge-related properties. The index list is sorted for the first entry.

    Args:
        edge_indices (str): Name of indices in dictionary. Default is "edge_indices".
        edge_attributes (str): Name of related edge values or attributes.
            This can be a match-string or list of names. Default is "^edge_.*".
    """
    def __init__(self, *, edge_indices: str = "edge_indices", edge_attributes: str = "^edge_.*",
                 name="sort_edge_indices", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"edge_indices": edge_indices, "edge_attributes": edge_attributes})
        self._to_assign = [edge_indices, edge_attributes]
        self._search = [edge_attributes]
        self._config_kwargs.update({"edge_indices": edge_indices, "edge_attributes": edge_attributes})

    def call(self, *, edge_indices: np.ndarray, edge_attributes: list):
        return sort_edge_indices(edge_indices, *edge_attributes, return_nested=True)


class SetEdgeIndicesReverse(GraphPreProcessorBase):
    r"""Computes the index map of the reverse edge for each of the edges, if available. This can be used by a model
    to directly select the corresponding edge of :math:`(j, i)` which is :math:`(i, j)`.
    Does not affect other edge-properties, only creates a map on edge indices. Edges that do not have a reverse
    pair get a `nan` as map index. If there are multiple edges, the first encounter is assigned.

    .. warning::
        Reverse maps are not recomputed if you use e.g. :obj:`sort_edge_indices` or redefine edges.

    Args:
        edge_indices (str): Name of indices in dictionary. Default is "edge_indices".
        edge_indices_reverse (str): Name of reverse indices to store output. Default is "edge_indices_reverse"
    """
    def __init__(self, *, edge_indices: str = "edge_indices", edge_indices_reverse: str = "edge_indices_reverse",
                 name="set_edge_indices_reverse", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"edge_indices": edge_indices})
        self._to_assign = edge_indices_reverse
        self._config_kwargs.update({"edge_indices": edge_indices, "edge_indices_reverse": edge_indices_reverse})

    def call(self, *, edge_indices: np.ndarray):
        if edge_indices is None:
            return None
        return np.expand_dims(compute_reverse_edges_index_map(edge_indices), axis=-1)


class PadProperty(GraphPreProcessorBase):
    r"""Pad a graph tensor property.

    Args:
        key (str): Name of the (tensor) property to pad.
        pad_width (list, int): Width to pad tensor.
        mode (str): Padding mode.
    """
    def __init__(self, *, key: str, pad_width: Union[int, list, tuple, np.ndarray], mode: str = "constant",
                 name="name", **kwargs):
        call_kwargs = {"pad_width": pad_width, "mode": mode}
        # List of additional kwargs for pad that do not belong to super.
        for x in ["stat_length", "constant_values", "end_values", "reflect_type"]:
            if x in kwargs:
                call_kwargs[x] = kwargs[x]
                kwargs.pop(x)
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"key": key})
        self._call_kwargs = call_kwargs
        self._to_assign = key
        self._config_kwargs.update({"key": key, **self._call_kwargs})

    def call(self, *, key: np.ndarray, pad_width: Union[int, list, tuple, np.ndarray], mode: str):
        if key is None:
            return
        return np.pad(key, pad_width=pad_width, mode=mode)


class SetEdgeWeightsUniform(GraphPreProcessorBase):
    r"""Adds or sets :obj:`edge_weights` with uniform values. Requires property :obj:`edge_indices`.
    Does not affect other edge-properties and only sets :obj:`edge_weights`.

    Args:
        edge_indices (str): Name of indices in dictionary. Default is "edge_indices".
        edge_weights (str): Name of edge weights to set in dictionary. Default is "edge_weights".
        value (float): Value to set :obj:`edge_weights` with. Default is 1.0.
    """

    def __init__(self, *, edge_indices: str = "edge_indices", edge_weights: str = "edge_weights",
                 value: float = 1.0, name="set_edge_weights_uniform", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"edge_indices": edge_indices})
        self._call_kwargs = {"value": value}
        self._to_assign = edge_weights
        self._config_kwargs.update({"edge_indices": edge_indices, "edge_weights": edge_weights, **self._call_kwargs})

    def call(self, *, edge_indices: np.ndarray, value: float):
        if edge_indices is None:
            return None
        return np.ones((len(edge_indices), 1)) * value


class NormalizeEdgeWeightsSymmetric(GraphPreProcessorBase):
    r"""Normalize :obj:`edge_weights` using the node degree of each row or column of the adjacency matrix.
    Normalize edge weights as :math:`\tilde{e}_{i,j} = d_{i,i}^{-0.5} \, e_{i,j} \, d_{j,j}^{-0.5}`.
    The node degree is defined as :math:`D_{i,i} = \sum_{j} A_{i, j}`. Requires the property :obj:`edge_indices`.
    Does not affect other edge-properties and only sets :obj:`edge_weights`.

    Args:
        edge_indices (str): Name of indices in dictionary. Default is "edge_indices".
        edge_weights (str): Name of edge weights indices to set in dictionary. Default is "edge_weights".
    """

    def __init__(self, *, edge_indices: str = "edge_indices", edge_weights: str = "edge_weights",
                 name="normalize_edge_weights_sym", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"edge_indices": edge_indices, "edge_weights": edge_weights})
        self._to_assign = edge_weights
        self._silent = edge_weights
        self._config_kwargs.update({"edge_indices": edge_indices, "edge_weights": edge_weights})

    def call(self, *, edge_indices: np.ndarray, edge_weights: np.ndarray):
        if edge_indices is None:
            return None
        # If weights is not set, initialize with weight one.
        if edge_weights is None:
            edge_weights = np.ones((len(edge_indices), 1))
        edge_weights = rescale_edge_weights_degree_sym(edge_indices, edge_weights)
        return edge_weights


class SetRangeFromEdges(GraphPreProcessorBase):
    r"""Assigns range indices and attributes (distance) from the definition of edge indices. These operations
    require the attributes :obj:`node_coordinates` and :obj:`edge_indices` to be set. That also means that
    :obj:`range_indices` will be equal to :obj:`edge_indices`.

    Args:
        edge_indices (str): Name of indices in dictionary. Default is "edge_indices".
        range_indices (str): Name of range indices to set in dictionary. Default is "range_indices".
        node_coordinates (str): Name of coordinates in dictionary. Default is "node_coordinates".
        range_attributes (str): Name of range distance to set in dictionary. Default is "range_attributes".
        do_invert_distance (bool): Invert distance when computing  :obj:`range_attributes`. Default is False.
    """
    def __init__(self, *, edge_indices: str = "edge_indices", range_indices: str = "range_indices",
                 node_coordinates: str = "node_coordinates", range_attributes: str = "range_attributes",
                 do_invert_distance: bool = False, name="set_range_from_edges", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"edge_indices": edge_indices, "node_coordinates": node_coordinates})
        self._call_kwargs = {"do_invert_distance": do_invert_distance}
        self._to_assign = [range_indices, range_attributes]
        self._config_kwargs.update({
            "edge_indices": edge_indices, "node_coordinates": node_coordinates, "range_indices": range_indices,
            "range_attributes": range_attributes, **self._call_kwargs})

    def call(self, *, edge_indices: np.ndarray, node_coordinates: np.ndarray, do_invert_distance: bool):
        if edge_indices is None:
            return None, None
        range_indices = np.array(edge_indices, dtype="int")  # Makes copy
        if node_coordinates is None:
            return range_indices, None
        dist = np.sqrt(np.sum(
            np.square(node_coordinates[range_indices[:, 0]] - node_coordinates[range_indices[:, 1]]),
            axis=-1, keepdims=True))
        if do_invert_distance:
            dist = invert_distance(dist)
        return range_indices, dist


class SetRange(GraphPreProcessorBase):
    r"""Define range in euclidean space for interaction or edge-like connections. The number of connection is
    determines based on a cutoff radius and a maximum number of neighbours or both.
    Requires :obj:`node_coordinates` to be set. The distance is stored in :obj:`range_attributes`.

    Args:
        range_indices (str): Name of range indices to set in dictionary. Default is "range_indices".
        node_coordinates (str): Name of coordinates in dictionary. Default is "node_coordinates".
        range_attributes (str): Name of range distance to set in dictionary. Default is "range_attributes".
        max_distance (float): Maximum distance or cutoff radius for connections. Default is 4.0.
        max_neighbours (int): Maximum number of allowed neighbours for a node. Default is 15.
        do_invert_distance (bool): Whether to invert the distance. Default is False.
        self_loops (bool): If also self-interactions with distance 0 should be considered. Default is False.
        exclusive (bool): Whether both max_neighbours and max_distance must be fulfilled. Default is True.
    """

    def __init__(self, *, range_indices: str = "range_indices", node_coordinates: str = "node_coordinates",
                 range_attributes: str = "range_attributes", max_distance: float = 4.0, max_neighbours: int = 15,
                 do_invert_distance: bool = False, self_loops: bool = False, exclusive: bool = True, name="set_range",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"node_coordinates": node_coordinates})
        self._call_kwargs = {
            "max_distance": max_distance, "max_neighbours": max_neighbours, "do_invert_distance": do_invert_distance,
            "self_loops": self_loops, "exclusive": exclusive}
        self._to_assign = [range_indices, range_attributes]
        self._config_kwargs.update({
            "node_coordinates": node_coordinates, "range_indices": range_indices, "range_attributes": range_attributes,
            **self._call_kwargs})

    def call(self, *, node_coordinates: np.ndarray, max_distance: float, max_neighbours: int, do_invert_distance: bool,
             self_loops: bool, exclusive: bool):
        if node_coordinates is None:
            return None, None
        # Compute distance matrix here. May be problematic for too large graphs.
        dist = coordinates_to_distancematrix(node_coordinates)
        cons, indices = define_adjacency_from_distance(
            dist, max_distance=max_distance, max_neighbours=max_neighbours, exclusive=exclusive, self_loops=self_loops)
        mask = np.array(cons, dtype="bool")
        dist_masked = dist[mask]
        if do_invert_distance:
            dist_masked = invert_distance(dist_masked)
        # Need one feature dimension.
        if len(dist_masked.shape) <= 1:
            dist_masked = np.expand_dims(dist_masked, axis=-1)
        # Assign attributes to instance.
        return indices, dist_masked


class SetAngle(GraphPreProcessorBase):
    r"""Find possible angles between geometric range connections defined by :obj:`range_indices`.
    Which edges form angles is stored in :obj:`angle_indices`.
    One can also change :obj:`range_indices` to `edge_indices` to compute angles between edges instead
    of range connections.

    .. warning::
        Angles are not recomputed if you use :obj:`set_range` or redefine range or edges.

    Args:
        range_indices (str): Name of range indices in dictionary. Default is "range_indices".
        node_coordinates (str): Name of coordinates in dictionary. Default is "node_coordinates".
        angle_indices (str): Name of angle (edge) indices to set in dictionary. Default is "angle_indices".
        angle_indices_nodes (str): Name of angle (node) indices to set in dictionary.
            Index triplets referring to nodes. Default is "angle_indices_nodes".
        angle_attributes (str): Name of angle values to set in dictionary. Default is "angle_attributes".
        allow_multi_edges (bool): Whether to allow angles between 'i<-j<-i', which gives 0-degree angle, if
            the nodes are unique. Default is False.
        compute_angles (bool): Whether to also compute angles.
    """
    def __init__(self, *, range_indices: str = "range_indices", node_coordinates: str = "node_coordinates",
                 angle_indices: str = "angle_indices", angle_indices_nodes: str = "angle_indices_nodes",
                 angle_attributes: str = "angle_attributes", allow_multi_edges: bool = False,
                 compute_angles: bool = True, name="set_angle", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"node_coordinates": node_coordinates, "range_indices": range_indices})
        self._call_kwargs = {"allow_multi_edges": allow_multi_edges, "compute_angles": compute_angles}
        self._to_assign = [angle_indices, angle_indices_nodes, angle_attributes]
        self._silent = node_coordinates
        self._config_kwargs.update({
            "node_coordinates": node_coordinates, "range_indices": range_indices, "angle_indices": angle_indices,
            "angle_indices_nodes": angle_indices_nodes, "angle_attributes": angle_attributes, **self._call_kwargs})

    def call(self, *, range_indices: np.ndarray, node_coordinates: np.ndarray,
             allow_multi_edges: bool, compute_angles: bool):
        if range_indices is None:
            return None, None, None
        # Compute Indices
        _, a_triples, a_indices = get_angle_indices(range_indices, allow_multi_edges=allow_multi_edges)
        # Also compute angles
        if compute_angles:
            if node_coordinates is not None:
                return a_indices, a_triples, get_angle(node_coordinates, a_triples)
        return a_indices, a_triples, None


class SetRangePeriodic(GraphPreProcessorBase):
    r"""Define range in euclidean space for interaction or edge-like connections on a periodic lattice.
    The number of connection is determines based on a cutoff radius and a maximum number of neighbours or both.
    Requires :obj:`node_coordinates`, :obj:`graph_lattice` to be set.
    The distance is stored in :obj:`range_attributes`.

    Args:
        range_indices (str): Name of range indices to set in dictionary. Default is "range_indices".
        node_coordinates (str): Name of coordinates in dictionary.
            Default is "node_coordinates".
        graph_lattice (str): Name of the lattice matrix. Default is "graph_lattice".
            The lattice vectors must be given in rows of the matrix!
        range_attributes (str): Name of range distance to set in dictionary. Default is "range_attributes".
        range_image (str): Name of range image indices to set in dictionary. Default is "range_image".
        max_distance (float): Maximum distance or cutoff radius for connections. Default is 4.0.
        max_neighbours (int, optional): Maximum number of allowed neighbours for each central atom. Default is None.
        exclusive (bool): Whether both distance and maximum neighbours must be fulfilled. Default is True.
        do_invert_distance (bool): Whether to invert the distance. Default is False.
        self_loops (bool): If also self-interactions with distance 0 should be considered. Default is False.
    """
    def __init__(self, *, range_indices: str = "range_indices", node_coordinates: str = "node_coordinates",
                 graph_lattice: str = "graph_lattice", range_image: str = "range_image",
                 range_attributes: str = "range_attributes", max_distance: float = 4.0, max_neighbours: int = None,
                 exclusive: bool = True, do_invert_distance: bool = False, self_loops: bool = False,
                 name="set_range_periodic", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"node_coordinates": node_coordinates, "graph_lattice": graph_lattice})
        self._call_kwargs = {
            "max_distance": max_distance, "max_neighbours": max_neighbours, "exclusive": exclusive,
            "do_invert_distance": do_invert_distance, "self_loops":self_loops}
        self._to_assign = [range_indices, range_image, range_attributes]
        self._config_kwargs.update({
            "node_coordinates": node_coordinates, "range_indices": range_indices, "graph_lattice": graph_lattice,
            "range_image": range_image, "range_attributes": range_attributes, **self._call_kwargs})

    def call(self, *, node_coordinates: np.ndarray, graph_lattice: np.ndarray, max_distance: float, max_neighbours: int,
             self_loops: bool, exclusive: bool, do_invert_distance: bool):
        if node_coordinates is None:
            return None, None, None
        if graph_lattice is None:
            return None, None, None

        indices, images, dist = range_neighbour_lattice(
            node_coordinates, graph_lattice,
            max_distance=max_distance, max_neighbours=max_neighbours, self_loops=self_loops, exclusive=exclusive)

        if do_invert_distance:
            dist = invert_distance(dist)
        # Need one feature dimension.
        if len(dist.shape) <= 1:
            dist = np.expand_dims(dist, axis=-1)
        # Assign attributes to instance.
        return indices, images, dist


def get():
    pass

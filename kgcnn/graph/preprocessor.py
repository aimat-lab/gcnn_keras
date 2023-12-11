import numpy as np
import logging
from typing import Union
from kgcnn.graph.base import GraphPreProcessorBase
from kgcnn.graph.methods import *

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
            This can be a match-string or list of names. Default is "^edge_(?!indices$).*".
        remove_duplicates (bool): Whether to remove duplicates within the new edges. Default is True.
        sort_indices (bool): Sort indices after adding edges. Default is True.
    """

    def __init__(self, edge_indices: str = "edge_indices", edge_attributes: str = "^edge_(?!indices$).*",
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
        if edge_indices is None:
            return None, [None] * len(edge_attributes)
        return add_edges_reverse_indices(
            edge_indices, *edge_attributes, remove_duplicates=remove_duplicates, sort_indices=sort_indices,
            return_nested=True)


class AddEdgeSelfLoops(GraphPreProcessorBase):
    r"""Add self loops to each graph property. The function expects the property :obj:`edge_indices`
    to be defined. By default, the edges are also sorted after adding the self-loops.
    All other edge properties are filled with :obj:`fill_value`.

    Args:
        edge_indices (str): Name of indices in dictionary. Default is "edge_indices".
        edge_attributes (str): Name of related edge values or attributes.
            This can be a match-string or list of names. Default is "^edge_(?!indices$).*".
        remove_duplicates (bool): Whether to remove duplicates. Default is True.
        sort_indices (bool): To sort indices after adding self-loops. Default is True.
        fill_value (in): The fill_value for all other edge properties.
    """

    def __init__(self, edge_indices: str = "edge_indices", edge_attributes: str = "^edge_(?!indices$).*",
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
        if edge_indices is None:
            return None, [None] * len(edge_attributes)
        return add_self_loops_to_edge_indices(
            edge_indices, *edge_attributes, remove_duplicates=remove_duplicates,
            sort_indices=sort_indices, fill_value=fill_value, return_nested=True)


class SortEdgeIndices(GraphPreProcessorBase):
    r"""Sort edge indices and all edge-related properties. The index list is sorted for the first entry.

    Args:
        edge_indices (str): Name of indices in dictionary. Default is "edge_indices".
        edge_attributes (str): Name of related edge values or attributes.
            This can be a match-string or list of names. Default is "^edge_(?!indices$).*".
    """

    def __init__(self, *, edge_indices: str = "edge_indices", edge_attributes: str = "^edge_(?!indices$).*",
                 name="sort_edge_indices", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"edge_indices": edge_indices, "edge_attributes": edge_attributes})
        self._to_assign = [edge_indices, edge_attributes]
        self._search = [edge_attributes]
        self._config_kwargs.update({"edge_indices": edge_indices, "edge_attributes": edge_attributes})

    def call(self, *, edge_indices: np.ndarray, edge_attributes: list):
        if edge_indices is None:
            return None, [None] * len(edge_attributes)
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
                 name="pad_property", **kwargs):
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
        self._silent = ["edge_weights"]
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
        overwrite (bool): Whether to overwrite existing range indices. Default is True.
    """

    def __init__(self, *, range_indices: str = "range_indices", node_coordinates: str = "node_coordinates",
                 range_attributes: str = "range_attributes", max_distance: float = 4.0, max_neighbours: int = 15,
                 do_invert_distance: bool = False, self_loops: bool = False, exclusive: bool = True, name="set_range",
                 overwrite: bool = True,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"node_coordinates": node_coordinates, "range_indices": range_indices})
        self._silent = ["range_indices"]
        self._call_kwargs = {
            "max_distance": max_distance, "max_neighbours": max_neighbours, "do_invert_distance": do_invert_distance,
            "self_loops": self_loops, "exclusive": exclusive, "overwrite": overwrite}
        self._to_assign = [range_indices, range_attributes]
        self._config_kwargs.update({
            "node_coordinates": node_coordinates, "range_indices": range_indices, "range_attributes": range_attributes,
            **self._call_kwargs})

    def call(self, *, node_coordinates: np.ndarray, range_indices: np.ndarray,
             max_distance: float, max_neighbours: int, do_invert_distance: bool,
             self_loops: bool, exclusive: bool, overwrite: bool):

        if range_indices is not None and not overwrite:
            # only need to recompute range_attributes.
            dist = distance_for_range_indices(coordinates=node_coordinates, indices=range_indices)
            if do_invert_distance:
                dist = invert_distance(dist)
            return range_indices, dist

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
                 angle_attributes: str = "angle_attributes",
                 allow_multi_edges: bool = False, allow_self_edges: bool = False, allow_reverse_edges: bool = False,
                 edge_pairing: str = "kj", check_sorted: bool = True,
                 compute_angles: bool = True, name="set_angle", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"node_coordinates": node_coordinates, "range_indices": range_indices})
        self._call_kwargs = {"allow_multi_edges": allow_multi_edges, "compute_angles": compute_angles,
                             "allow_self_edges": allow_self_edges, "edge_pairing": edge_pairing,
                             "allow_reverse_edges": allow_reverse_edges, "check_sorted": check_sorted}
        self._to_assign = [angle_indices, angle_indices_nodes, angle_attributes]
        self._silent = ["node_coordinates"]
        self._config_kwargs.update({
            "node_coordinates": node_coordinates, "range_indices": range_indices, "angle_indices": angle_indices,
            "angle_indices_nodes": angle_indices_nodes, "angle_attributes": angle_attributes, **self._call_kwargs})

    def call(self, *, range_indices: np.ndarray, node_coordinates: np.ndarray,
             check_sorted: bool, allow_multi_edges: bool,
             allow_self_edges: bool, allow_reverse_edges: bool,
             edge_pairing: str, compute_angles: bool):
        if range_indices is None:
            return None, None, None
        # Compute Indices
        _, a_triples, a_indices = get_angle_indices(
            range_indices, allow_multi_edges=allow_multi_edges, allow_self_edges=allow_self_edges,
            allow_reverse_edges=allow_reverse_edges, check_sorted=check_sorted, edge_pairing=edge_pairing)
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
        overwrite (bool): Whether to overwrite existing range indices. Default is True.
    """

    def __init__(self, *, range_indices: str = "range_indices", node_coordinates: str = "node_coordinates",
                 graph_lattice: str = "graph_lattice", range_image: str = "range_image",
                 range_attributes: str = "range_attributes", max_distance: float = 4.0, max_neighbours: int = None,
                 exclusive: bool = True, do_invert_distance: bool = False, self_loops: bool = False,
                 overwrite: bool = True,
                 name="set_range_periodic", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({
            "node_coordinates": node_coordinates, "graph_lattice": graph_lattice, "range_indices": range_indices,
            "range_image": range_image})
        self._silent = ["range_indices", "range_image"]
        self._call_kwargs = {
            "max_distance": max_distance, "max_neighbours": max_neighbours, "exclusive": exclusive,
            "do_invert_distance": do_invert_distance, "self_loops": self_loops, "overwrite": overwrite}
        self._to_assign = [range_indices, range_image, range_attributes]
        self._config_kwargs.update({
            "node_coordinates": node_coordinates, "range_indices": range_indices, "graph_lattice": graph_lattice,
            "range_image": range_image, "range_attributes": range_attributes,
            **self._call_kwargs})

    def call(self, *, node_coordinates: np.ndarray, graph_lattice: np.ndarray,
             range_indices: np.ndarray, range_image: np.ndarray,
             max_distance: float, max_neighbours: int,
             self_loops: bool, exclusive: bool, do_invert_distance: bool,
             overwrite: bool) -> tuple:
        if all([range_item is not None for range_item in
                [range_indices, range_image]]) and not overwrite:
            # need to recompute range_attributes.
            dist = distance_for_range_indices_periodic(
                coordinates=node_coordinates, indices=range_indices, lattice=graph_lattice, images=range_image)
            if do_invert_distance:
                dist = invert_distance(dist)
            return range_indices, range_image, dist
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


class ExpandDistanceGaussianBasis(GraphPreProcessorBase):
    r"""Expand distance into Gaussian basis a features or attributes.

    Args:
        range_attributes (str): Name of distances in dictionary. Default is "range_attributes".
        bins (int): Number of bins to sample distance from. Default is 20.
        distance (value): Maximum distance to be captured by bins. Default is 4.0.
        sigma (value): Sigma of the Gaussian function, determining the width/sharpness. Default is 0.4.
        offset (float): Possible offset to center Gaussian. Default is 0.0.
        axis (int): Axis to expand distance. Defaults to -1.
        expand_dims (bool): Whether to expand dims. Default to True.
    """

    def __init__(self, *, range_attributes: str = "range_attributes",
                 bins: int = 20, distance: float = 4.0, sigma: float = 0.4, offset: float = 0.0, axis: int = -1,
                 expand_dims: bool = True,
                 name="expand_distance_gaussian_basis", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"range_attributes": range_attributes})
        self._to_assign = range_attributes
        self._call_kwargs = {"bins": bins, "distance": distance, "sigma": sigma, "offset": offset, "axis": axis,
                             "expand_dims": expand_dims}
        self._config_kwargs.update({"range_attributes": range_attributes, **self._call_kwargs})

    def call(self, *, range_attributes: np.ndarray,
             bins: int, distance: float, sigma: float, offset: float, axis: int, expand_dims: bool):
        if range_attributes is None:
            return None
        return distance_to_gauss_basis(range_attributes, bins=bins, distance=distance, sigma=sigma, offset=offset,
                                       axis=axis, expand_dims=expand_dims)


class AtomicChargesRepresentation(GraphPreProcessorBase):
    r"""Generate a node representation based on atomic charges.
    Multiplies a one-hot representation of charges with multiple powers of the (scaled) charge itself.
    This has been used by `EGNN <https://arxiv.org/abs/2102.09844>`_.

    Args:
        node_number (str): Name of atomic charges in dictionary. Default is "node_number".
        node_attributes (str): Name of related attributes to assign output. Defaults to "node_attributes".
        one_hot (list): List of possible charges. Default is [1, 6, 7, 8, 9].
        charge_scale (float): Scale by which the charges are scaled for power. Default is 9.0.
        charge_power (int): Maximum power up to which compute charge powers. Default is 2.
    """

    def __init__(self, *, node_number: str = "node_number", node_attributes: str = "node_attributes",
                 name="atomic_charge_representation", one_hot: list = None, charge_scale: float = 9.0,
                 charge_power: int = 2, **kwargs):
        super().__init__(name=name, **kwargs)
        if one_hot is None:
            one_hot = [1, 6, 7, 8, 9]
        self._to_obtain.update({"node_number": node_number})
        self._to_assign = node_attributes
        self._call_kwargs = {"one_hot": one_hot, "charge_scale": charge_scale, "charge_power": charge_power}
        self._config_kwargs.update({"node_number": node_number, "node_attributes": node_attributes,
                                    **self._call_kwargs})

    def call(self, *, node_number: np.ndarray, one_hot: list, charge_scale: float, charge_power: int):
        if node_number is None:
            return None
        if len(node_number.shape) <= 1:
            node_number = np.expand_dims(node_number, axis=-1)
        oh = np.array(node_number == np.array([one_hot], dtype="int"), dtype="float")  # (N, ohe)
        charge_tensor = np.power(node_number / charge_scale, np.arange(charge_power + 1))  # (N, power)
        atom_scalars = np.expand_dims(oh, axis=-1) * np.expand_dims(charge_tensor, axis=1)  # (N, ohe, power)
        return atom_scalars.reshape((len(node_number), -1))


class PrincipalMomentsOfInertia(GraphPreProcessorBase):
    r"""Store the principle moments of the matrix of inertia for a set of node coordinates and masses into
    a graph property defined by :obj:`graph_inertia`.

    Args:
        node_mass (str): Name of node mass in dictionary. Default is "node_mass".
        node_coordinates (str): Name of node coordinates. Defaults to "node_coordinates".
        graph_inertia (str): Name of output property to store moments. Default is "graph_inertia".
        shift_center_of_mass (bool): Whether to shift to center of mass. Default is True.
    """

    def __init__(self, *, node_mass: str = "node_mass", node_coordinates: str = "node_coordinates",
                 graph_inertia: str = "graph_inertia", name="principal_moments_of_inertia",
                 shift_center_of_mass: bool = True, **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"node_mass": node_mass, "node_coordinates": node_coordinates})
        self._to_assign = graph_inertia
        self._call_kwargs = {"shift_center_of_mass": shift_center_of_mass}
        self._config_kwargs.update({"node_mass": node_mass, "node_coordinates": node_coordinates,
                                    "graph_inertia": graph_inertia, **self._call_kwargs})

    def call(self, *, node_mass: np.ndarray, node_coordinates: np.ndarray, shift_center_of_mass: bool):
        if node_mass is None or node_coordinates is None:
            return None
        return get_principal_moments_of_inertia(
            masses=node_mass, coordinates=node_coordinates, shift_center_of_mass=shift_center_of_mass)


class ShiftToUnitCell(GraphPreProcessorBase):
    r"""Shift atomic coordinates into the Unit cell of a periodic lattice.

    Args:
        node_coordinates (str): Name of node coordinates. Defaults to "node_coordinates".
        graph_lattice (str): Name of the graph lattice. Defaults to "graph_lattice".
    """

    def __init__(self, *, node_coordinates: str = "node_coordinates", graph_lattice: str = "graph_lattice",
                 name="shift_to_unit_cell", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"node_coordinates": node_coordinates, "graph_lattice": graph_lattice})
        self._to_assign = node_coordinates
        self._call_kwargs = {}
        self._config_kwargs.update({"node_coordinates": node_coordinates, "graph_lattice": graph_lattice,
                                    **self._call_kwargs})

    def call(self, *, node_coordinates: np.ndarray, graph_lattice: np.ndarray):
        if node_coordinates is None or graph_lattice is None:
            return None
        return shift_coordinates_to_unit_cell(coordinates=node_coordinates, lattice=graph_lattice)


class CountNodesAndEdges(GraphPreProcessorBase):
    r"""Count the number of nodes and edges.

    Args:
        node_attributes (str): Name for nodes attributes to count nodes.
        edge_indices (str): Name for edge_indices to count edges.
        count_node (str): Name to assign node count to.
        count_edge (str): Name to assign edge count to.
    """

    def __init__(self, *, total_nodes: str = "total_nodes", total_edges: str = "total_edges",
                 count_nodes: str = "node_attributes", count_edges: str = "edge_indices",
                 name="count_nodes_and_edges", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"count_nodes": count_nodes, "count_edges": count_edges})
        self._to_assign = [total_nodes, total_edges]
        self._config_kwargs.update({"total_nodes": total_nodes, "total_edges": total_edges,
                                    "count_nodes": count_nodes, "count_edges": count_edges})

    def call(self, *, count_nodes: np.ndarray, count_edges: np.ndarray):
        total_nodes = len(count_nodes) if count_nodes is not None else None
        total_edges = len(count_edges) if count_edges is not None else None
        return total_nodes, total_edges


class MakeDenseAdjacencyMatrix(GraphPreProcessorBase):
    r"""Make adjacency matrix based on edge indices.

    Args:
        edge_indices (str): Name of adjacency matrix.
        edge_attributes (str): Name of edge attributes.
        adjacency_matrix (str): Name of adjacency matrix to assign output.
    """

    def __init__(self, *, edge_indices: str = "edge_indices", edge_attributes: str = "edge_attributes",
                 adjacency_matrix: str = "adjacency_matrix",
                 name="make_dense_adjacency_matrix", **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"edge_indices": edge_indices, "edge_attributes": edge_attributes})
        self._to_assign = adjacency_matrix
        self._config_kwargs.update({"adjacency_matrix": adjacency_matrix, "edge_indices": edge_indices,
                                    "edge_attributes": edge_attributes})

    def call(self, *, edge_indices: np.ndarray, edge_attributes: np.ndarray):
        if edge_indices is None or edge_attributes is None:
            return None
        if len(edge_indices) == 0:
            return None
        max_index = np.amax(edge_indices)+1
        adj = np.zeros([max_index, max_index] + list(edge_attributes.shape[1:]))
        adj[edge_indices[:, 0], edge_indices[:, 1]] = edge_attributes
        return adj


class MakeMask(GraphPreProcessorBase):
    r"""Make simple equally sized (dummy) mask for a graph property.

    Args:
        target_property (str): Name of the property to make mask for.
        mask_name (str): Name of mask to assign output.
        rank (int): Rank of mask.
    """

    def __init__(self, *, target_property: str = "node_attributes", mask_name: str = None,
                 rank=1,
                 name="make_mask", **kwargs):
        super().__init__(name=name, **kwargs)
        if mask_name is None:
            mask_name = target_property+"_mask"
        self._to_obtain.update({"target_property": target_property})
        self._to_assign = mask_name
        self._call_kwargs = {"rank": rank}
        self._config_kwargs.update({"mask_name": mask_name, "target_property": target_property, "rank": rank})

    def call(self, *, target_property: np.ndarray, rank: int):
        if target_property is None:
            return None
        mask = np.ones(target_property.shape[:rank], dtype="bool")
        return mask

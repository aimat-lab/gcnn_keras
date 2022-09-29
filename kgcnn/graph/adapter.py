import logging
import numpy as np
import functools
import re
from typing import Union

from kgcnn.graph.adj import get_angle_indices, coordinates_to_distancematrix, invert_distance, \
    define_adjacency_from_distance, sort_edge_indices, get_angle, add_edges_reverse_indices, \
    rescale_edge_weights_degree_sym, add_self_loops_to_edge_indices, compute_reverse_edges_index_map
from kgcnn.graph.geom import range_neighbour_lattice

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

module_logger.info(
    "`GraphMethodsAdapter` is deprecated and will be removed in future versions in favor of `GraphPreProcessorBase`. \
This is done, in order not to pollute `GraphDict`s namespace for increasing number of methods.")


def obtain_assign_properties(obtain: Union[list, str] = None, assign: Union[list, str] = None,
                             silent: Union[list, str] = None):
    """Decorator to obtain and assign properties to container from function input and output.

    Args:
       obtain:
       assign:
       silent:
    """
    if obtain is None:
        obtain = []
    if assign is None:
        assign = []
    if silent is None:
        silent = []
    if not isinstance(obtain, (list, tuple)):
        obtain = [obtain]
    if not isinstance(silent, (list, tuple)):
        silent = [silent]
    assign_is_list = isinstance(assign, (list, tuple))
    if not assign_is_list:
        assign = [assign]

    def has_special_characters(s, pat=re.compile("[@\\\\!#$%^&*()<>?/|}{~:]")):
        return pat.search(s) is not None

    def function_obtain_property(self, key, quiet):
        if isinstance(key, str):
            if not self.has_valid_key(key) and not quiet:
                module_logger.warning("Missing '%s' in '%s'" % (key, type(self).__name__))
            return self.obtain_property(key)
        return key

    def function_assign_property(self, key, value):
        self.assign_property(key, value)

    def obtain_assign_decorator(func):

        @functools.wraps(func)
        def function_wrapper(self, *args, **kwargs):
            properties_to_obtain = [
                [x, kwargs[x] if x in kwargs else x, True if x in silent else False] for x in obtain]
            properties_to_assign = [kwargs[x] if x in kwargs else x for x in assign]

            obtained_properties = {
                key: function_obtain_property(self, value, verbose) for key, value, verbose in properties_to_obtain}
            other_kwargs = {key: value for key, value in kwargs.items() if key not in obtain}

            if len(args) > 0:
                module_logger.warning("`GraphTensorMethodsAdapter` can only fetch properties in kwargs, not %s" % args)

            output = func(self, *args, **obtained_properties, **other_kwargs)
            if not assign_is_list:
                output = [output]

            for key, value in zip(properties_to_assign, output):
                function_assign_property(self, key, value)
            return self

        return function_wrapper

    return obtain_assign_decorator


class GraphTensorMethodsAdapter:

    def search_properties(self, keys: str) -> list:
        raise NotImplementedError("Must be implemented by container class")

    def has_valid_key(self, key: str) -> bool:
        raise NotImplementedError("Must be implemented by container class")

    def obtain_property(self, key: str) -> Union[np.ndarray, None]:
        raise NotImplementedError("Must be implemented by container class")

    def assign_property(self, key: str, value: np.ndarray):
        raise NotImplementedError("Must be implemented by container class")

    def _operate_on_edges(
            self, operation, edge_indices: str = "edge_indices", edge_attributes: str = "^edge_.*",
            **kwargs):
        r"""General wrapper to run a certain function on an array of edge-indices and all edge related properties
        in the dictionary of self.
        The name of the key for indices must be defined in :obj:`edge_indices`. Related value or attribute tensors
        are searched by :obj:`edge_attributes`, since they must behave accordingly.

        Args:
              operation (callable): Function to apply to a list of all edge arrays.
                First entry is assured to be indices.
              edge_indices (str): Name of indices in dictionary. Default is "edge_indices".
              edge_attributes (str): Name of related edge values or attributes.
                This can be a match-string or list of names. Default is "^edge_.*".
              kwargs: Kwargs for operation function call.

        Returns:
            self
        """
        if not self.has_valid_key(edge_indices):
            module_logger.error("Can not operate on edges, missing '%s'." % edge_indices)
            return self
        # Determine all linked edge attributes, that are not None.
        edge_linked = self.search_properties(edge_attributes)
        # Edge indices is always at first position!
        edge_linked = [edge_indices] + [x for x in edge_linked if x != edge_indices]
        no_nan_edge_prop = [x for x in edge_linked if self.obtain_property(x) is not None]
        non_nan_edge = [self.obtain_property(x) for x in no_nan_edge_prop]
        new_edges = operation(*non_nan_edge, **kwargs)
        # If dataset only has edge indices, operation is expected to only return array not list!
        # This restricts the type of operation used with this method.
        if len(no_nan_edge_prop) == 1:
            new_edges = [new_edges]
        # Set all new edge attributes.
        for at, value in zip(no_nan_edge_prop, new_edges):
            self.assign_property(at, value)
        return self

    def make_undirected_edges(self, edge_indices: str = "edge_indices", edge_attributes: str = "^edge_.*",
                              remove_duplicates: bool = True, sort_indices: bool = True):
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

        Returns:
            self
        """
        self._operate_on_edges(add_edges_reverse_indices, edge_indices=edge_indices,
                               edge_attributes=edge_attributes,
                               remove_duplicates=remove_duplicates, sort_indices=sort_indices)
        return self

    def add_edge_self_loops(self, edge_indices: str = "edge_indices",
                            edge_attributes: str = "^edge_.*",
                            remove_duplicates: bool = True,
                            sort_indices: bool = True,
                            fill_value: int = 0):
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

        Returns:
            self
        """
        if not self.has_valid_key(edge_indices):
            module_logger.error("Can not set 'add_edge_self_loops', missing '%s'." % edge_indices)
            return self
        self._operate_on_edges(add_self_loops_to_edge_indices, edge_indices=edge_indices,
                               edge_attributes=edge_attributes,
                               remove_duplicates=remove_duplicates, sort_indices=sort_indices, fill_value=fill_value)
        return self

    def sort_edge_indices(self, edge_indices: str = "edge_indices",
                          edge_attributes: str = "^edge_.*"):
        r"""Sort edge indices and all edge-related properties. The index list is sorted for the first entry.

        Args:
            edge_indices (str): Name of indices in dictionary. Default is "edge_indices".
            edge_attributes (str): Name of related edge values or attributes.
                This can be a match-string or list of names. Default is "^edge_.*".

        Returns:
            self
        """
        if not self.has_valid_key(edge_indices):
            module_logger.error("Can not set 'sort_edge_indices', missing '%s'." % edge_indices)
            return self
        self._operate_on_edges(sort_edge_indices, edge_indices=edge_indices,
                               edge_attributes=edge_attributes)
        return self

    @obtain_assign_properties(obtain="edge_indices", assign="edge_indices_reverse")
    def set_edge_indices_reverse(self, *, edge_indices: Union[str, np.ndarray] = "edge_indices",
                                 edge_indices_reverse: str = "edge_indices_reverse"):
        r"""Computes the index map of the reverse edge for each of the edges, if available. This can be used by a model
        to directly select the corresponding edge of :math:`(j, i)` which is :math:`(i, j)`.
        Does not affect other edge-properties, only creates a map on edge indices. Edges that do not have a reverse
        pair get a `nan` as map index. If there are multiple edges, the first encounter is assigned.

        .. warning::
            Reverse maps are not recomputed if you use e.g. :obj:`sort_edge_indices` or redefine edges.

        Args:
            edge_indices (str): Name of indices in dictionary. Default is "edge_indices".
            edge_indices_reverse (str): Name of reverse indices to store output. Default is "edge_indices_reverse"

        Returns:
            self
        """
        if edge_indices is None:
            return None
        return np.expand_dims(compute_reverse_edges_index_map(edge_indices), axis=-1)

    @obtain_assign_properties(obtain="key", assign="key")
    def pad_property(self, *, key: Union[str, np.ndarray],
                     pad_width: Union[int, list, np.ndarray], mode: str = "constant", **kwargs):
        r"""Pad a graph tensor property.

        Args:
            key (str): Name of the (tensor) property to pad.
            pad_width (list, int): Width to pad tensor.
            mode (str): Padding mode.

        Returns:
            self
        """
        if key is None:
            return
        return np.pad(key, pad_width=pad_width, mode=mode, **kwargs)

    @obtain_assign_properties(obtain=["edge_indices"], assign="edge_weights")
    def set_edge_weights_uniform(self, *, edge_indices: Union[str, np.ndarray] = "edge_indices",
                                 edge_weights: str = "edge_weights", value: float = 1.0):
        r"""Adds or sets :obj:`edge_weights` with. Requires the property :obj:`edge_indices`.
        Does not affect other edge-properties and only sets :obj:`edge_weights`.

        Args:
            edge_indices (str): Name of indices in dictionary. Default is "edge_indices".
            edge_weights (str): Name of edge weights to set in dictionary. Default is "edge_weights".
            value (float): Value to set :obj:`edge_weights` with. Default is 1.0.

        Returns:
            self
        """
        if edge_indices is None:
            return None
        return np.ones((len(edge_indices), 1)) * value

    @obtain_assign_properties(obtain=["edge_indices", "edge_weights"], assign="edge_weights", silent="edge_weights")
    def normalize_edge_weights_sym(self, *, edge_indices: Union[str, np.ndarray] = "edge_indices",
                                   edge_weights: str = "edge_weights"):
        r"""Normalize :obj:`edge_weights` using the node degree of each row or column of the adjacency matrix.
        Normalize edge weights as :math:`\tilde{e}_{i,j} = d_{i,i}^{-0.5} \, e_{i,j} \, d_{j,j}^{-0.5}`.
        The node degree is defined as :math:`D_{i,i} = \sum_{j} A_{i, j}`. Requires the property :obj:`edge_indices`.
        Does not affect other edge-properties and only sets :obj:`edge_weights`.

        Args:
            edge_indices (str): Name of indices in dictionary. Default is "edge_indices".
            edge_weights (str): Name of edge weights indices to set in dictionary. Default is "edge_weights".

        Returns:
            self
        """
        if edge_indices is None:
            return None
        # If weights is not set, initialize with weight one.
        if edge_weights is None:
            edge_weights = np.ones((len(edge_indices), 1))
        edge_weights = rescale_edge_weights_degree_sym(edge_indices, edge_weights)
        return edge_weights

    @obtain_assign_properties(obtain=["edge_indices", "node_coordinates"], assign=["range_indices", "range_attributes"])
    def set_range_from_edges(self, *, edge_indices: Union[str, np.ndarray] = "edge_indices",
                             range_indices: str = "range_indices",
                             node_coordinates: Union[str, np.ndarray] = "node_coordinates",
                             range_attributes: str = "range_attributes",
                             do_invert_distance: bool = False):
        r"""Assigns range indices and attributes (distance) from the definition of edge indices. These operations
        require the attributes :obj:`node_coordinates` and :obj:`edge_indices` to be set. That also means that
        :obj:`range_indices` will be equal to :obj:`edge_indices`.

        Args:
            edge_indices (str): Name of indices in dictionary. Default is "edge_indices".
            range_indices (str): Name of range indices to set in dictionary. Default is "range_indices".
            node_coordinates (str): Name of coordinates in dictionary. Default is "node_coordinates".
            range_attributes (str): Name of range distance to set in dictionary. Default is "range_attributes".
            do_invert_distance (bool): Invert distance when computing  :obj:`range_attributes`. Default is False.

        Returns:
            self
        """
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

    @obtain_assign_properties(obtain="node_coordinates", assign=["range_indices", "range_attributes"])
    def set_range(self, *, range_indices: str = "range_indices",
                  node_coordinates: Union[str, np.ndarray] = "node_coordinates",
                  range_attributes: str = "range_attributes",
                  max_distance: float = 4.0, max_neighbours: int = 15,
                  do_invert_distance: bool = False, self_loops: bool = False, exclusive: bool = True):
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

        Returns:
            self
        """
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

    @obtain_assign_properties(obtain=["node_coordinates", "range_indices"], silent="node_coordinates",
                              assign=["angle_indices", "angle_indices_nodes", "angle_attributes"])
    def set_angle(self, range_indices: Union[str, np.ndarray] = "range_indices",
                  node_coordinates: Union[str, np.ndarray] = "node_coordinates",
                  angle_indices: str = "angle_indices",
                  angle_indices_nodes: str = "angle_indices_nodes",
                  angle_attributes: str = "angle_attributes",
                  allow_multi_edges: bool = False,
                  compute_angles: bool = True):
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

        Returns:
            self
        """
        if range_indices is None:
            return None, None, None
        # Compute angles
        _, a_triples, a_indices = get_angle_indices(range_indices, allow_multi_edges=allow_multi_edges)
        # Also compute angles
        if compute_angles:
            if node_coordinates is not None:
                return a_indices, a_triples, get_angle(node_coordinates, a_triples)
        return a_indices, a_triples, None

    @obtain_assign_properties(obtain=["node_coordinates", "graph_lattice"],
                              assign=["range_indices", "range_image", "range_attributes"])
    def set_range_periodic(self, range_indices: str = "range_indices",
                           node_coordinates: Union[str, np.ndarray] = "node_coordinates",
                           graph_lattice: Union[str, np.ndarray] = "graph_lattice",
                           range_image: str = "range_image",
                           range_attributes: str = "range_attributes",
                           max_distance: float = 4.0, max_neighbours: int = None,
                           exclusive: bool = True,
                           do_invert_distance: bool = False, self_loops: bool = False):
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

        Returns:
            self
        """
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


GraphMethodsAdapter = GraphTensorMethodsAdapter

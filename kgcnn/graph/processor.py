import numpy as np
import logging
from kgcnn.graph.base import GraphPreProcessorBase, GraphDict
from kgcnn.graph.adj import get_angle_indices, coordinates_to_distancematrix, invert_distance, \
    define_adjacency_from_distance, sort_edge_indices, get_angle, add_edges_reverse_indices, \
    rescale_edge_weights_degree_sym, add_self_loops_to_edge_indices, compute_reverse_edges_index_map
from kgcnn.graph.geom import range_neighbour_lattice

logging.basicConfig()  # Module logger
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


class MakeUndirectedEdges(GraphPreProcessorBase):
    def __init__(self, name="make_undirected_edges", **kwargs):
        super().__init__(name=name, **kwargs)


class SetEdgeWeightsUniform(GraphPreProcessorBase):

    def __init__(self, *, edge_indices: str = "edge_indices", edge_weights: str = "edge_weights",
                 value: float = 1.0, name="set_edge_weights_uniform", **kwargs):
        r"""Adds or sets :obj:`edge_weights` with uniform values. Requires property :obj:`edge_indices`.
        Does not affect other edge-properties and only sets :obj:`edge_weights`.

        Args:
            edge_indices (str): Name of indices in dictionary. Default is "edge_indices".
            edge_weights (str): Name of edge weights to set in dictionary. Default is "edge_weights".
            value (float): Value to set :obj:`edge_weights` with. Default is 1.0.
        """
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"edge_indices": edge_indices})
        self._config_kwargs.update({"value": value, "edge_indices": edge_indices, "edge_weights": edge_weights})
        self._call_kwargs = {"value": value}
        self._to_assign = edge_weights

    def call(self, edge_indices: np.ndarray, value: float):
        if edge_indices is None:
            return None
        return np.ones((len(edge_indices), 1)) * value


class SetRange(GraphPreProcessorBase):

    def __init__(self, *, range_indices: str = "range_indices", node_coordinates: str = "node_coordinates",
                 range_attributes: str = "range_attributes", max_distance: float = 4.0, max_neighbours: int = 15,
                 do_invert_distance: bool = False, self_loops: bool = False, exclusive: bool = True, name="set_range",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self._to_obtain.update({"node_coordinates": node_coordinates})
        self._config_kwargs.update({
            "node_coordinates": node_coordinates, "range_indices": range_indices, "range_attributes":range_attributes,
            "max_distance": max_distance, "max_neighbours": max_neighbours, "do_invert_distance": do_invert_distance,
            "self_loops": self_loops, "exclusive": exclusive})
        self._call_kwargs = {
            "max_distance": max_distance, "max_neighbours": max_neighbours, "do_invert_distance": do_invert_distance,
            "self_loops": self_loops, "exclusive": exclusive}
        self._to_assign = [range_indices, range_attributes]

    def call(self, node_coordinates: np.ndarray, max_distance: float, max_neighbours: int,
             do_invert_distance: bool, self_loops: bool, exclusive: bool):
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


def get():
    pass

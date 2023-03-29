import tensorflow as tf
from kgcnn.layers.geom import FracToRealCoordinates, VectorAngle

from kgcnn.literature.coGN._graph_network.graph_network_base import GraphNetworkBase


class EdgeDisplacementVectorDecoder(GraphNetworkBase):
    r"""This layer calculates edge offsets (displacement vectors between two nodes) based on node coordinates features.

    The layer has different modes to work on molecular graphs
    as well as crystal unit cell graphs and crystal asymmetric unit graphs. 

    For molecular graphs set :python:`periodic_boundary_condition` to :python:`False`.
    Node features must be the coordinates of each node.

    For crystal unit cell graphs set :python:`periodic_boundary_condition` to :python:`True`.
    Node features must be the fractional coordinates of each node.
    Edge features must be the cell translation for each edge in the unit cell graph.
    Global features must be the lattice matrices for each crystal graph.

    For crystal asymmetric unit graphs set :python:`symmetrized` to :python:`True`.
    Node features must be the fractional coordinates of each node.
    Edge features must be a tuple/list of the cell translation and affine matrix for each edge.
    Global features must be the lattice matrices for each crystal graph.

    The purpose of this layer is to make GNNs differentiable with respect to atom (fractional) coordinates.
    """

    def __init__(self, periodic_boundary_condition=False, symmetrized=False, **kwargs):
        super().__init__(
            aggregate_edges_local=None,
            aggregate_edges_global=None,
            aggregate_nodes=None,
            **kwargs
        )
        # symmetrized -> periodic_boundary_condition
        if symmetrized:
            periodic_boundary_condition = True

        self.symmetrized = symmetrized
        self.periodic_boundary_condition = periodic_boundary_condition
        self.frac_to_real_layer = FracToRealCoordinates()

    def update_edges(
        self, edge_features, nodes_in, nodes_out, global_features, **kwargs
    ):
        in_coords = nodes_in.values
        out_coords = nodes_out.values
        lattice_matrices = global_features

        if self.periodic_boundary_condition:
            if self.symmetrized:
                cell_translations, affine_matrices = edge_features

                # Affine Transformation
                out_coords_ = tf.concat(
                    [
                        out_coords,
                        tf.expand_dims(tf.ones_like(out_coords[:, 0]), axis=1),
                    ],
                    axis=1,
                )
                out_coords = tf.einsum(
                    "ij,ikj->ik", out_coords_, affine_matrices.values
                )[:, :-1]
                out_coords = out_coords - tf.floor(
                    out_coords
                )  # All values should be in [0,1) interval
            else:
                cell_translations = edge_features

            # Cell translation
            out_coords = out_coords + cell_translations.values

        offset = in_coords - out_coords
        offset = tf.RaggedTensor.from_row_splits(
            offset, nodes_in.row_splits, validate=self.ragged_validate
        )

        if self.periodic_boundary_condition:
            offset = self.frac_to_real_layer([offset, lattice_matrices])

        return offset


class LineGraphAngleDecoder(GraphNetworkBase):
    """Compute edge angles from line graph node features (offset vectors)."""

    def __init__(self, **kwargs):
        super().__init__(
            aggregate_edges_local=None,
            aggregate_edges_global=None,
            aggregate_nodes=None,
            **kwargs
        )
        self.vector_angle_layer = VectorAngle()

    def update_edges(
        self, edge_features, nodes_in, nodes_out, global_features, **kwargs
    ):
        angles = self.vector_angle_layer([nodes_in, nodes_out])[:, :, 0]
        return angles

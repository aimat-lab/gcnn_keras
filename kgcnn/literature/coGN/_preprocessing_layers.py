import tensorflow as tf
from kgcnn.layers.geom import FracToRealCoordinates, VectorAngle

from kgcnn.literature.coGN._graph_network.graph_network_base import GraphNetworkBase


class EdgeDisplacementVectorDecoder(GraphNetworkBase):
    """Layer to make GNNs differentiable with respect to atom (fractional) coordinates for crystal graphs."""

    def __init__(self, symmetrized=False, **kwargs):
        super().__init__(
            aggregate_edges_local=None,
            aggregate_edges_global=None,
            aggregate_nodes=None,
            **kwargs
        )
        self.symmetrized = symmetrized
        self.frac_to_real_layer = FracToRealCoordinates()

    def update_edges(
        self, edge_features, nodes_in, nodes_out, global_features, **kwargs
    ):
        in_frac_coords = nodes_in.values
        out_frac_coords = nodes_out.values
        lattice_matrices = global_features

        if self.symmetrized:
            cell_translations, affine_matrices = edge_features

            # Affine Transformation
            out_frac_coords_ = tf.concat(
                [
                    out_frac_coords,
                    tf.expand_dims(tf.ones_like(out_frac_coords[:, 0]), axis=1),
                ],
                axis=1,
            )
            out_frac_coords = tf.einsum(
                "ij,ikj->ik", out_frac_coords_, affine_matrices.values
            )[:, :-1]
            out_frac_coords = out_frac_coords - tf.floor(
                out_frac_coords
            )  # All values should be in [0,1) interval
        else:
            cell_translations = edge_features

        # Cell translation
        out_frac_coords = out_frac_coords + cell_translations.values

        offset = in_frac_coords - out_frac_coords
        offset = tf.RaggedTensor.from_row_splits(
            offset, cell_translations.row_splits, validate=self.ragged_validate
        )

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

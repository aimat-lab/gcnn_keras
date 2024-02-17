import keras as ks
import kgcnn.ops.activ
from kgcnn.layers.geom import (
    DisplacementVectorsUnitCell,
    DisplacementVectorsASU, NodePosition, FracToRealCoordinates,
    EuclideanNorm, GaussBasisLayer
)
from kgcnn.layers.mlp import MLP
from kgcnn.layers.modules import Embedding
from kgcnn.layers.pooling import PoolingWeightedNodes, PoolingNodes
from ._layers import CGCNNLayer


def model_disjoint_crystal(
        inputs: list,
        representation=None,
        use_node_embedding=None,
        output_embedding=None,
        input_node_embedding=None,
        expand_distance=None,
        conv_layer_args=None,
        make_distances=None,
        depth=None,
        gauss_args=None,
        node_pooling_args=None,
        output_mlp=None
):
    atom_attributes, multiplicities, coord, symmops, edge_indices, cell_translations, lattice_matrix, batch_id_node, batch_id_edge, count_nodes, count_edges = inputs

    if make_distances:

        frac_coords = coord

        if representation == 'unit':
            displacement_vectors = DisplacementVectorsUnitCell()([frac_coords, edge_indices, cell_translations])

        elif representation == 'asu':
            displacement_vectors = DisplacementVectorsASU()([frac_coords, edge_indices, symmops, cell_translations])
        else:
            x_in, x_out = NodePosition()([frac_coords, edge_indices])
            displacement_vectors = ks.layers.Subtract()([x_out, x_in])

        displacement_vectors = FracToRealCoordinates()([displacement_vectors, lattice_matrix, batch_id_edge])

        edge_distances = EuclideanNorm(axis=-1, keepdims=True)(displacement_vectors)

    else:
        edge_distances = coord

    if expand_distance:
        edge_distances = GaussBasisLayer(**gauss_args)(edge_distances)

    # embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(atom_attributes)
    else:
        n = atom_attributes

    n = ks.layers.Dense(conv_layer_args["units"], activation='linear')(n)
    for _ in range(depth):
        n = CGCNNLayer(**conv_layer_args)([
            n, edge_distances, edge_indices, batch_id_node, batch_id_edge, count_nodes, count_edges
        ])

    if representation == 'asu':
        out = PoolingWeightedNodes(**node_pooling_args)([count_nodes, n, multiplicities, batch_id_node])
    else:
        out = PoolingNodes(**node_pooling_args)([count_nodes, n, batch_id_node])

    out = MLP(**output_mlp)(out)

    # Only graph embedding for CGCNN.
    if output_embedding != "graph":
        raise ValueError("Unsupported output embedding for mode `CGCNN` .")

    return out

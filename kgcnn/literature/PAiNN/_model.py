from keras.layers import Add
from kgcnn.layers.geom import NodePosition, EdgeDirectionNormalized, NodeDistanceEuclidean, CosCutOffEnvelope, \
    BesselBasisLayer, ShiftPeriodicLattice
from kgcnn.layers.mlp import MLP, GraphMLP
from kgcnn.layers.modules import Embedding
from kgcnn.layers.norm import GraphLayerNormalization, GraphBatchNormalization
from kgcnn.layers.pooling import PoolingNodes
from ._layers import EquivariantInitialize, PAiNNconv, PAiNNUpdate


def model_disjoint(
        inputs: list,
        use_node_embedding: bool,
        input_node_embedding: dict,
        equiv_initialize_kwargs: dict,
        bessel_basis: dict,
        depth: int,
        pooling_args: dict,
        conv_args: dict,
        update_args: dict,
        equiv_normalization: bool,
        node_normalization: bool,
        output_embedding: str,
        output_mlp: dict,
):
    z, x, edi, batch_id_node, batch_id_edge, count_nodes, count_edges, v = inputs

    if v is None:
        v = EquivariantInitialize(**equiv_initialize_kwargs)(z)

    # Optional Embedding.
    if use_node_embedding:
        z = Embedding(**input_node_embedding)(z)

    pos1, pos2 = NodePosition()([x, edi])
    rij = EdgeDirectionNormalized()([pos1, pos2])
    d = NodeDistanceEuclidean()([pos1, pos2])
    env = CosCutOffEnvelope(conv_args["cutoff"])(d)
    rbf = BesselBasisLayer(**bessel_basis)(d)

    for i in range(depth):
        # Message
        ds, dv = PAiNNconv(**conv_args)([z, v, rbf, env, rij, edi])
        z = Add()([z, ds])
        v = Add()([v, dv])
        # Update
        ds, dv = PAiNNUpdate(**update_args)([z, v])
        z = Add()([z, ds])
        v = Add()([v, dv])

        if equiv_normalization:
            v = GraphLayerNormalization(axis=2)([v, batch_id_edge, count_edges])
        if node_normalization:
            z = GraphBatchNormalization(axis=-1)([z, batch_id_node, count_nodes])

    n = z
    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes(**pooling_args)([count_nodes, n, batch_id_node])
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `PAiNN`")

    return out


def model_disjoint_crystal(
        inputs: list,
        use_node_embedding: bool,
        input_node_embedding: dict,
        equiv_initialize_kwargs: dict,
        bessel_basis: dict,
        depth: int,
        pooling_args: dict,
        conv_args: dict,
        update_args: dict,
        equiv_normalization: bool,
        node_normalization: bool,
        output_embedding: str,
        output_mlp: dict,
):
    z, x, edi, edge_image, lattice, batch_id_node, batch_id_edge, count_nodes, count_edges, v = inputs

    if v is None:
        v = EquivariantInitialize(**equiv_initialize_kwargs)(z)

    # Optional Embedding.
    if use_node_embedding:
        z = Embedding(**input_node_embedding)(z)

    pos1, pos2 = NodePosition()([x, edi])
    pos2 = ShiftPeriodicLattice()([pos2, edge_image, lattice, batch_id_edge])
    rij = EdgeDirectionNormalized()([pos1, pos2])
    d = NodeDistanceEuclidean()([pos1, pos2])
    env = CosCutOffEnvelope(conv_args["cutoff"])(d)
    rbf = BesselBasisLayer(**bessel_basis)(d)

    for i in range(depth):
        # Message
        ds, dv = PAiNNconv(**conv_args)([z, v, rbf, env, rij, edi])
        z = Add()([z, ds])
        v = Add()([v, dv])
        # Update
        ds, dv = PAiNNUpdate(**update_args)([z, v])
        z = Add()([z, ds])
        v = Add()([v, dv])

        if equiv_normalization:
            v = GraphLayerNormalization(axis=2)([v, batch_id_edge, count_edges])
        if node_normalization:
            z = GraphBatchNormalization(axis=-1)([z, batch_id_node, count_nodes])

    n = z
    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes(**pooling_args)([count_nodes, n, batch_id_node])
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `PAiNN`")

    return out
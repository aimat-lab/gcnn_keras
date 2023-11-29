from keras.layers import Dense
from kgcnn.layers.conv import SchNetInteraction
from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean, GaussBasisLayer, ShiftPeriodicLattice
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.modules import Embedding
from kgcnn.layers.pooling import PoolingNodes


def model_disjoint(
        inputs,
        use_node_embedding: bool = None,
        input_node_embedding: dict = None,
        make_distance: bool = None,
        expand_distance: bool = None,
        gauss_args: dict = None,
        interaction_args: dict = None,
        node_pooling_args: dict = None,
        depth: int = None,
        last_mlp: dict = None,
        output_embedding: str = None,
        use_output_mlp: bool = None,
        output_mlp: dict = None):
    n, x, disjoint_indices, batch_id_node, count_nodes = inputs

    # Optional Embedding.
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)

    if make_distance:
        pos1, pos2 = NodePosition()([x, disjoint_indices])
        ed = NodeDistanceEuclidean()([pos1, pos2])
    else:
        ed = x

    if expand_distance:
        ed = GaussBasisLayer(**gauss_args)(ed)

    # Model
    n = Dense(interaction_args["units"], activation='linear')(n)
    for i in range(0, depth):
        n = SchNetInteraction(**interaction_args)([n, ed, disjoint_indices])

    n = GraphMLP(**last_mlp)([n, batch_id_node, count_nodes])

    # Output embedding choice
    if output_embedding == 'graph':
        out = PoolingNodes(**node_pooling_args)([count_nodes, n, batch_id_node])
        if use_output_mlp:
            out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = n
        if use_output_mlp:
            out = GraphMLP(**output_mlp)([out, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `SchNet` .")

    return out


def model_disjoint_crystal(
        inputs,
        use_node_embedding: bool = None,
        input_node_embedding: dict = None,
        make_distance: bool = None,
        expand_distance: bool = None,
        gauss_args: dict = None,
        interaction_args: dict = None,
        node_pooling_args: dict = None,
        depth: int = None,
        last_mlp: dict = None,
        output_embedding: str = None,
        use_output_mlp: bool = None,
        output_mlp: dict = None):
    n, x, disjoint_indices, edge_image, lattice, batch_id_node, batch_id_edge, count_nodes = inputs

    # Optional Embedding.
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)

    if make_distance:
        pos1, pos2 = NodePosition()([x, disjoint_indices])
        pos2 = ShiftPeriodicLattice()([pos2, edge_image, lattice, batch_id_edge])
        ed = NodeDistanceEuclidean()([pos1, pos2])
    else:
        ed, _, _, _ = x

    if expand_distance:
        ed = GaussBasisLayer(**gauss_args)(ed)

    # Model
    n = Dense(interaction_args["units"], activation='linear')(n)
    for i in range(0, depth):
        n = SchNetInteraction(**interaction_args)([n, ed, disjoint_indices])

    n = GraphMLP(**last_mlp)([n, batch_id_node, count_nodes])

    # Output embedding choice
    if output_embedding == 'graph':
        out = PoolingNodes(**node_pooling_args)([count_nodes, n, batch_id_node])
        if use_output_mlp:
            out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = n
        if use_output_mlp:
            out = GraphMLP(**output_mlp)([out, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `SchNet` .")

    return out

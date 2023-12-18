import keras as ks
from ._layers import TrafoEdgeNetMessages
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing
from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean, GaussBasisLayer, ShiftPeriodicLattice
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.modules import Embedding
from kgcnn.layers.message import MatMulMessages
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.set2set import PoolingSet2SetEncoder


def model_disjoint(inputs,
                   use_node_embedding: bool = None,
                   use_edge_embedding: bool = None,
                   input_node_embedding: dict = None,
                   input_edge_embedding: dict = None,
                   geometric_edge: bool = None,
                   make_distance: bool = None,
                   expand_distance: bool = None,
                   gauss_args: dict = None,
                   set2set_args: dict = None,
                   pooling_args: dict = None,
                   edge_mlp: dict = None,
                   use_set2set: bool = None,
                   node_dim: int = None,
                   depth: int = None,
                   output_embedding: str = None,
                   output_mlp: dict = None):
    n0, ed, disjoint_indices, batch_id_node, batch_id_edge, count_nodes, count_edges = inputs

    # embedding, if no feature dimension
    if use_node_embedding:
        n0 = Embedding(**input_node_embedding)(n0)

    if not geometric_edge:
        if use_edge_embedding:
            ed = Embedding(**input_edge_embedding)(ed)

    if make_distance:
        pos1, pos2 = NodePosition()([ed, disjoint_indices])
        ed = NodeDistanceEuclidean()([pos1, pos2])

    if expand_distance:
        ed = GaussBasisLayer(**gauss_args)(ed)

    # Make hidden dimension
    n = ks.layers.Dense(node_dim, activation="linear")(n0)

    # Make edge networks.
    edge_net_in = GraphMLP(**edge_mlp)([ed, batch_id_edge, count_edges])
    edge_net_in = TrafoEdgeNetMessages(target_shape=(node_dim, node_dim))(edge_net_in)
    edge_net_out = GraphMLP(**edge_mlp)([ed, batch_id_edge, count_edges])
    edge_net_out = TrafoEdgeNetMessages(target_shape=(node_dim, node_dim))(edge_net_out)

    # Gru for node updates
    gru = GRUUpdate(node_dim)

    for i in range(0, depth):
        n_in = GatherNodesOutgoing()([n, disjoint_indices])
        n_out = GatherNodesIngoing()([n, disjoint_indices])
        m_in = MatMulMessages()([edge_net_in, n_in])
        m_out = MatMulMessages()([edge_net_out, n_out])
        eu = ks.layers.Concatenate(axis=-1)([m_in, m_out])
        eu = AggregateLocalEdges(**pooling_args)([n, eu, disjoint_indices])  # Summing for each node connections
        n = gru([n, eu])

    n = ks.layers.Concatenate(axis=-1)([n0, n])

    # Output embedding choice
    if output_embedding == 'graph':
        if use_set2set:
            # output
            n = ks.layers.Dense(units=set2set_args['channels'], activation="linear")(n)
            out = PoolingSet2SetEncoder(**set2set_args)([count_nodes, n, batch_id_node])
        else:
            out = PoolingNodes(**pooling_args)([count_nodes, n, batch_id_node])
        out = ks.layers.Flatten()(out)  # Flatten() required for to Set2Set output.
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `NMPN` .")

    return out


def model_disjoint_crystal(inputs,
                           use_node_embedding: bool = None,
                           use_edge_embedding: bool = None,
                           input_node_embedding: dict = None,
                           input_edge_embedding: dict = None,
                           geometric_edge: bool = None,
                           make_distance: bool = None,
                           expand_distance: bool = None,
                           gauss_args: dict = None,
                           set2set_args: dict = None,
                           pooling_args: dict = None,
                           edge_mlp: dict = None,
                           use_set2set: bool = None,
                           node_dim: int = None,
                           depth: int = None,
                           output_embedding: str = None,
                           output_mlp: dict = None):
    n0, ed, disjoint_indices, edge_image, lattice, batch_id_node, batch_id_edge, count_nodes, count_edges = inputs

    # embedding, if no feature dimension
    if use_node_embedding:
        n0 = Embedding(**input_node_embedding)(n0)

    if not geometric_edge:
        if use_edge_embedding:
            ed = Embedding(**input_edge_embedding)(ed)

    # If coordinates are in place of edges
    if make_distance:
        x = ed
        pos1, pos2 = NodePosition()([x, disjoint_indices])
        pos2 = ShiftPeriodicLattice()([pos2, edge_image, lattice, batch_id_edge])
        ed = NodeDistanceEuclidean()([pos1, pos2])

    if expand_distance:
        ed = GaussBasisLayer(**gauss_args)(ed)

    # Make hidden dimension
    n = ks.layers.Dense(node_dim, activation="linear")(n0)

    # Make edge networks.
    edge_net_in = GraphMLP(**edge_mlp)([ed, batch_id_edge, count_edges])
    edge_net_in = TrafoEdgeNetMessages(target_shape=(node_dim, node_dim))(edge_net_in)
    edge_net_out = GraphMLP(**edge_mlp)([ed, batch_id_edge, count_edges])
    edge_net_out = TrafoEdgeNetMessages(target_shape=(node_dim, node_dim))(edge_net_out)

    # Gru for node updates
    gru = GRUUpdate(node_dim)

    for i in range(0, depth):
        n_in = GatherNodesOutgoing()([n, disjoint_indices])
        n_out = GatherNodesIngoing()([n, disjoint_indices])
        m_in = MatMulMessages()([edge_net_in, n_in])
        m_out = MatMulMessages()([edge_net_out, n_out])
        eu = ks.layers.Concatenate(axis=-1)([m_in, m_out])
        eu = AggregateLocalEdges(**pooling_args)([n, eu, disjoint_indices])  # Summing for each node connections
        n = gru([n, eu])

    n = ks.layers.Concatenate(axis=-1)([n0, n])

    # Output embedding choice
    if output_embedding == 'graph':
        if use_set2set:
            # output
            n = ks.layers.Dense(units=set2set_args['channels'], activation="linear")(n)
            out = PoolingSet2SetEncoder(**set2set_args)([count_nodes, n, batch_id_node])
        else:
            out = PoolingNodes(**pooling_args)([count_nodes, n, batch_id_node])
        out = ks.layers.Flatten()(out)  # Flatten() required for to Set2Set output.
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `NMPN` .")

    return out

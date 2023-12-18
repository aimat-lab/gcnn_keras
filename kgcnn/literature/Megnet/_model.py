from kgcnn.layers.modules import Embedding
from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean, GaussBasisLayer, ShiftPeriodicLattice
from kgcnn.layers.mlp import MLP, GraphMLP
from keras.layers import Dense, Dropout, Concatenate, Flatten, Add
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.set2set import PoolingSet2SetEncoder
from ._layers import MEGnetBlock


PoolingGlobalEdges = PoolingNodes


def model_disjoint(
        inputs,
        use_node_embedding,
        use_graph_embedding,
        input_node_embedding: dict = None,
        input_graph_embedding: dict = None,
        expand_distance: bool = None,
        make_distance: bool = None,
        gauss_args: dict = None,
        meg_block_args: dict = None,
        set2set_args: dict = None,
        node_ff_args: dict = None,
        edge_ff_args: dict = None,
        state_ff_args: dict = None,
        use_set2set: bool = None,
        nblocks: int = None,
        has_ff: bool = None,
        dropout: float = None,
        output_embedding: str = None,
        output_mlp: dict = None,
):
    # Make input
    vp, x, edi, up, batch_id_node, batch_id_edge, count_nodes, count_edges = inputs

    # Embedding, if no feature dimension
    if use_node_embedding:
        vp = Embedding(**input_node_embedding)(vp)
    if use_graph_embedding:
        up = Embedding(**input_graph_embedding)(up)

    # Edge distance as Gauss-Basis
    if make_distance:
        pos1, pos2 = NodePosition()([x, edi])
        ep = NodeDistanceEuclidean()([pos1, pos2])
    else:
        ep = x

    if expand_distance:
        ep = GaussBasisLayer(**gauss_args)(ep)

    # Model
    vp = GraphMLP(**node_ff_args)([vp, batch_id_node, count_nodes])
    ep = GraphMLP(**edge_ff_args)([ep, batch_id_edge, count_edges])
    up = MLP(**state_ff_args)(up)
    vp2 = vp
    ep2 = ep
    up2 = up
    for i in range(0, nblocks):
        if has_ff and i > 0:
            vp2 = GraphMLP(**node_ff_args)([vp, batch_id_node, count_nodes])
            ep2 = GraphMLP(**edge_ff_args)([ep, batch_id_edge, count_edges])
            up2 = MLP(**state_ff_args)(up)

        # MEGnetBlock
        vp2, ep2, up2 = MEGnetBlock(**meg_block_args)(
            [vp2, ep2, edi, up2, batch_id_node, batch_id_edge, count_nodes, count_edges])

        # skip connection
        if dropout is not None:
            vp2 = Dropout(dropout, name='dropout_atom_%d' % i)(vp2)
            ep2 = Dropout(dropout, name='dropout_bond_%d' % i)(ep2)
            up2 = Dropout(dropout, name='dropout_state_%d' % i)(up2)

        vp = Add()([vp2, vp])
        ep = Add()([ep2, ep])
        up = Add()([up2, up])

    if use_set2set:
        vp = Dense(set2set_args["channels"], activation='linear')(vp)  # to match units
        ep = Dense(set2set_args["channels"], activation='linear')(ep)  # to match units
        vp = PoolingSet2SetEncoder(**set2set_args)([count_nodes, vp, batch_id_node])
        ep = PoolingSet2SetEncoder(**set2set_args)([count_edges, ep, batch_id_edge])
    else:
        vp = PoolingNodes()([count_nodes, vp, batch_id_node])
        ep = PoolingGlobalEdges()([count_edges, ep, batch_id_edge])

    ep = Flatten()(ep)
    vp = Flatten()(vp)
    final_vec = Concatenate(axis=-1)([vp, ep, up])

    if dropout is not None:
        final_vec = Dropout(dropout, name='dropout_final')(final_vec)

    # Only graph embedding for Megnet
    if output_embedding != "graph":
        raise ValueError("Unsupported output embedding for mode `Megnet`.")

    main_output = MLP(**output_mlp)(final_vec)
    return main_output


def model_disjoint_crystal(
        inputs,
        use_node_embedding,
        use_graph_embedding,
        input_node_embedding: dict = None,
        input_graph_embedding: dict = None,
        expand_distance: bool = None,
        make_distance: bool = None,
        gauss_args: dict = None,
        meg_block_args: dict = None,
        set2set_args: dict = None,
        node_ff_args: dict = None,
        edge_ff_args: dict = None,
        state_ff_args: dict = None,
        use_set2set: bool = None,
        nblocks: int = None,
        has_ff: bool = None,
        dropout: float = None,
        output_embedding: str = None,
        output_mlp: dict = None,
):
    vp, x, edi, up, edge_image, lattice, batch_id_node, batch_id_edge, count_nodes, count_edges = inputs

    # Embedding, if no feature dimension
    if use_node_embedding:
        vp = Embedding(**input_node_embedding)(vp)
    if use_graph_embedding:
        up = Embedding(**input_graph_embedding)(up)

    # Edge distance as Gauss-Basis
    if make_distance:
        pos1, pos2 = NodePosition()([x, edi])
        pos2 = ShiftPeriodicLattice()([pos2, edge_image, lattice, batch_id_edge])
        ep = NodeDistanceEuclidean()([pos1, pos2])
    else:
        ep = x

    if expand_distance:
        ep = GaussBasisLayer(**gauss_args)(ep)

    # Model
    vp = GraphMLP(**node_ff_args)([vp, batch_id_edge, count_edges])
    ep = GraphMLP(**edge_ff_args)([ep, batch_id_edge, count_edges])
    up = MLP(**state_ff_args)(up)
    vp2 = vp
    ep2 = ep
    up2 = up
    for i in range(0, nblocks):
        if has_ff and i > 0:
            vp2 = GraphMLP(**node_ff_args)([vp, batch_id_node, count_nodes])
            ep2 = GraphMLP(**edge_ff_args)([ep, batch_id_edge, count_edges])
            up2 = MLP(**state_ff_args)(up)

        # MEGnetBlock
        vp2, ep2, up2 = MEGnetBlock(**meg_block_args)(
            [vp2, ep2, edi, up2, batch_id_node, batch_id_edge, count_nodes, count_edges])

        # skip connection
        if dropout is not None:
            vp2 = Dropout(dropout, name='dropout_atom_%d' % i)(vp2)
            ep2 = Dropout(dropout, name='dropout_bond_%d' % i)(ep2)
            up2 = Dropout(dropout, name='dropout_state_%d' % i)(up2)

        vp = Add()([vp2, vp])
        ep = Add()([ep2, ep])
        up = Add()([up2, up])

    if use_set2set:
        vp = Dense(set2set_args["channels"], activation='linear')(vp)  # to match units
        ep = Dense(set2set_args["channels"], activation='linear')(ep)  # to match units
        vp = PoolingSet2SetEncoder(**set2set_args)([count_nodes, vp, batch_id_node])
        ep = PoolingSet2SetEncoder(**set2set_args)([count_edges, ep, batch_id_edge])
    else:
        vp = PoolingNodes()([count_nodes, vp, batch_id_node])
        ep = PoolingGlobalEdges()([count_edges, ep, batch_id_edge])

    ep = Flatten()(ep)
    vp = Flatten()(vp)
    final_vec = Concatenate(axis=-1)([vp, ep, up])

    if dropout is not None:
        final_vec = Dropout(dropout, name='dropout_final')(final_vec)

    # Only graph embedding for Megnet
    if output_embedding != "graph":
        raise ValueError("Unsupported output embedding for mode `Megnet`.")

    main_output = MLP(**output_mlp)(final_vec)

    return main_output

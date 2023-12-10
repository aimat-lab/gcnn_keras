from kgcnn.layers.modules import Embedding
from kgcnn.layers.gather import GatherState, GatherNodesIngoing, GatherNodesOutgoing
from keras.layers import Concatenate, Dense, Flatten
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.set2set import PoolingSet2SetEncoder
from kgcnn.layers.mlp import MLP, GraphMLP


def model_disjoint(
        inputs,
        use_node_embedding,
        use_edge_embedding,
        use_graph_embedding,
        input_node_embedding,
        input_edge_embedding,
        input_graph_embedding,
        gather_args,
        depth,
        edge_mlp_args,
        pooling_args,
        node_mlp_args,
        output_embedding,
        use_set2set,
        set2set_args,
        output_mlp):
    # Make input
    n, ed, edi, uenv, batch_id_node, batch_id_edge, count_nodes, count_edges = inputs

    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)
    if use_edge_embedding:
        ed = Embedding(**input_edge_embedding)(ed)
    if use_graph_embedding:
        uenv = Embedding(**input_graph_embedding)(uenv)

    # Model
    ev = GatherState(**gather_args)([uenv, batch_id_node])
    # n-Layer Step
    for i in range(0, depth):
        # upd = GatherNodes()([n,edi])
        eu1 = GatherNodesIngoing(**gather_args)([n, edi])
        eu2 = GatherNodesOutgoing(**gather_args)([n, edi])
        upd = Concatenate(axis=-1)([eu2, eu1])
        eu = Concatenate(axis=-1)([upd, ed])

        eu = GraphMLP(**edge_mlp_args)([eu, batch_id_edge, count_edges])
        # Pool message
        nu = AggregateLocalEdges(**pooling_args)([n, eu, edi])  # Summing for each node connection
        # Add environment
        nu = Concatenate(axis=-1)(
            [n, nu, ev])  # LazyConcatenate node features with new edge updates
        n = GraphMLP(**node_mlp_args)([nu, batch_id_node, count_nodes])

    # Output embedding choice
    if output_embedding == 'graph':
        if use_set2set:
            # output
            n = Dense(set2set_args["channels"], activation="linear")(n)
            out = PoolingSet2SetEncoder(**set2set_args)([count_nodes, n, batch_id_node])
        else:
            out = PoolingNodes(**pooling_args)([count_nodes, n, batch_id_node])
        out = Flatten()(out)
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `INorp` .")

    return out

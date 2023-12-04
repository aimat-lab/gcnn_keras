from keras.layers import Dense
from kgcnn.layers.conv import GCN
from kgcnn.layers.mlp import MLP, GraphMLP
from kgcnn.layers.modules import Embedding
from kgcnn.layers.pooling import PoolingNodes, PoolingWeightedNodes

# from keras_core.layers import Activation
# from kgcnn.layers.aggr import AggregateWeightedLocalEdges
# from kgcnn.layers.gather import GatherNodesOutgoing


def model_disjoint(inputs,
                   use_node_embedding: bool = None,
                   use_edge_embedding: bool = None,
                   input_node_embedding: dict = None,
                   input_edge_embedding: dict = None,
                   depth: int = None,
                   gcn_args: dict = None,
                   node_pooling_args: dict = None,
                   output_embedding: str = None,
                   output_mlp: dict = None,
                   ):
    n, e, disjoint_indices, batch_id_node, count_nodes = inputs

    # Embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)
    if use_edge_embedding:
        e = Embedding(**input_edge_embedding)(e)

    # Model
    n = Dense(gcn_args["units"], use_bias=True, activation='linear')(n)  # Map to units
    for i in range(0, depth):
        n = GCN(**gcn_args)([n, e, disjoint_indices])

        # # Equivalent as:
        # no = Dense(gcn_args["units"], activation="linear")(n)
        # no = GatherNodesOutgoing()([no, disjoint_indices])
        # nu = AggregateWeightedLocalEdges()([n, no, disjoint_indices, e])
        # n = Activation(gcn_args["activation"])(nu)

    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes(**node_pooling_args)([count_nodes, n, batch_id_node])  # will return tensor
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for `GCN` .")

    return out


def model_disjoint_weighted(inputs,
                            use_node_embedding: bool = None,
                            use_edge_embedding: bool = None,
                            input_node_embedding: dict = None,
                            input_edge_embedding: dict = None,
                            depth: int = None,
                            gcn_args: dict = None,
                            node_pooling_args: dict = None,
                            output_embedding: str = None,
                            output_mlp: dict = None,
                            ):
    n, nw, e, disjoint_indices, batch_id_node, count_nodes = inputs
    # Embedding, if no feature dimension

    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)
    if use_edge_embedding:
        e = Embedding(**input_edge_embedding)(e)

    # Model
    n = Dense(gcn_args["units"], use_bias=True, activation='linear')(n)  # Map to units
    for i in range(0, depth):
        n = GCN(**gcn_args)([n, e, disjoint_indices])

    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingWeightedNodes(**node_pooling_args)([count_nodes, n, nw, batch_id_node])  # will return tensor
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for `GCN`")

    return out

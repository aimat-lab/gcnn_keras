from keras.layers import Dense, Add, Multiply, Activation
from kgcnn.layers.modules import Embedding
from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.relational import RelationalDense
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.mlp import MLP, GraphMLP


def model_disjoint(
        inputs,
        use_node_embedding,
        use_edge_embedding,
        input_node_embedding=None,
        input_edge_embedding=None,
        depth=None,
        dense_kwargs=None,
        dense_relation_kwargs=None,
        activation_kwargs=None,
        node_pooling_kwargs=None,
        output_mlp=None,
        output_embedding=None
):
    n, edge_weights, edge_relations, edi, batch_id_node, count_nodes = inputs

    # Embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)
    if use_edge_embedding:
        edge_weights = Embedding(**input_edge_embedding)(edge_weights)

    # Model
    for i in range(0, depth):
        n_j = GatherNodesOutgoing()([n, edi])
        h0 = Dense(**dense_kwargs)(n)
        h_j = RelationalDense(**dense_relation_kwargs)([n_j, edge_relations])
        m = Multiply()([h_j, edge_weights])
        h = AggregateLocalEdges(pooling_method="sum")([n, m, edi])
        n = Add()([h, h0])
        n = Activation(**activation_kwargs)(n)

    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes(**node_pooling_kwargs)([count_nodes, n, batch_id_node])  # will return tensor
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":  # Node labeling
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `RGCN`")

    return out

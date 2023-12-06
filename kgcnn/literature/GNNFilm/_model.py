from keras.layers import Dense, Add, Multiply, Activation
from kgcnn.layers.modules import Embedding
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.relational import RelationalDense
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.mlp import MLP, GraphMLP


def model_disjoint(
        inputs,
        use_node_embedding,
        input_node_embedding=None,
        depth=None,
        dense_modulation_kwargs=None,
        dense_relation_kwargs=None,
        activation_kwargs=None,
        output_embedding=None,
        output_mlp=None,
        node_pooling_kwargs=None
):
    n, edge_relations, edi, batch_id_node, count_nodes = inputs

    # Embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)

    # Model
    for i in range(0, depth):
        n_i, n_j = GatherNodes(split_indices=[0, 1], concat_axis=None)([n, edi])
        # Note: This maybe could be done more efficiently.
        gamma = RelationalDense(**dense_modulation_kwargs)([n_i, edge_relations])
        beta = RelationalDense(**dense_modulation_kwargs)([n_i, edge_relations])
        h_j = RelationalDense(**dense_relation_kwargs)([n_j, edge_relations])
        m = Multiply()([h_j, gamma])
        m = Add()([m, beta])
        h = AggregateLocalEdges(pooling_method="sum")([n, m, edi])
        n = Activation(**activation_kwargs)(h)

    # Output embedding choice
    if output_embedding == "graph":
        out = PoolingNodes(**node_pooling_kwargs)([count_nodes, n, batch_id_node])  # will return tensor
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":  # Node labeling
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `GNNFilm`")

    return out

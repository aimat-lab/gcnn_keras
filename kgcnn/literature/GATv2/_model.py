from keras.layers import Dense, Concatenate, Average, Activation
from kgcnn.layers.attention import AttentionHeadGATV2
from kgcnn.layers.mlp import MLP, GraphMLP
from kgcnn.layers.modules import Embedding
from kgcnn.layers.pooling import PoolingNodes


def model_disjoint(inputs,
                   use_node_embedding: bool = None,
                   use_edge_embedding: bool = None,
                   input_node_embedding: dict = None,
                   input_edge_embedding: dict = None,
                   attention_args: dict = None,
                   pooling_nodes_args: dict = None,
                   depth: int = None,
                   attention_heads_num: int = None,
                   attention_heads_concat: bool = None,
                   output_embedding: str = None,
                   output_mlp: dict = None,
                   ):
    n, ed, disjoint_indices, batch_id_node, count_nodes = inputs

    # Embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)
    if use_edge_embedding:
        ed = Embedding(**input_edge_embedding)(ed)

    # Model
    nk = Dense(units=attention_args["units"], activation="linear")(n)
    for i in range(0, depth):
        heads = [AttentionHeadGATV2(**attention_args)([nk, ed, disjoint_indices]) for _ in range(attention_heads_num)]
        if attention_heads_concat:
            nk = Concatenate(axis=-1)(heads)
        else:
            nk = Average()(heads)
            nk = Activation(activation=attention_args["activation"])(nk)
    n = nk

    # Output embedding choice
    if output_embedding == 'graph':
        out = PoolingNodes(**pooling_nodes_args)([count_nodes, n, batch_id_node])
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for `GATv2` .")

    return out
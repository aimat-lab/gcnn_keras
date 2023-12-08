from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.modules import Embedding
from keras.layers import Dense, Dropout, Add
from kgcnn.layers.pooling import PoolingNodes
from ._layers import rGIN


def model_disjoint(
    inputs,
    use_node_embedding,
    input_node_embedding,
    gin_mlp,
    depth,
    rgin_args,
    last_mlp,
    output_mlp,
    output_embedding,
    dropout
):
    n, edi, batch_id_node, count_nodes = inputs

    # Embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)

    # Model
    # Map to the required number of units.
    n_units = gin_mlp["units"][-1] if isinstance(gin_mlp["units"], list) else int(gin_mlp["units"])
    n = Dense(n_units, use_bias=True, activation='linear')(n)
    list_embeddings = [n]
    for i in range(0, depth):
        n = rGIN(**rgin_args)([n, edi])
        n = GraphMLP(**gin_mlp)([n, batch_id_node, count_nodes])
        list_embeddings.append(n)

    # Output embedding choice
    if output_embedding == "graph":
        out = [PoolingNodes()([count_nodes, x, batch_id_node]) for x in list_embeddings]  # will return tensor
        out = [MLP(**last_mlp)(x) for x in out]
        out = [Dropout(dropout)(x) for x in out]
        out = Add()(out)
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":  # Node labeling
        out = n
        out = GraphMLP(**last_mlp)([out, batch_id_node, count_nodes])
        out = GraphMLP(**output_mlp)([out, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `rGIN`")

    return out

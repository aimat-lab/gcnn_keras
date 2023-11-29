import keras as ks
from keras.layers import Dense
from kgcnn.layers.conv import GIN, GINE
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.modules import Embedding
from kgcnn.layers.pooling import PoolingNodes


def model_disjoint(inputs,
                   use_node_embedding: bool = None,
                   input_node_embedding: dict = None,
                   depth: int = None,
                   gin_args: dict = None,
                   gin_mlp: dict = None,
                   last_mlp: dict = None,
                   dropout: float = None,
                   output_embedding: str = None,
                   output_mlp: dict = None):
    n, disjoint_indices, batch_id_node, count_nodes = inputs

    # Embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)

    # Model
    # Map to the required number of units.
    n_units = gin_mlp["units"][-1] if isinstance(gin_mlp["units"], list) else int(gin_mlp["units"])
    n = Dense(n_units, use_bias=True, activation='linear')(n)
    list_embeddings = [n]
    for i in range(0, depth):
        n = GIN(**gin_args)([n, disjoint_indices])
        n = GraphMLP(**gin_mlp)([n, batch_id_node, count_nodes])
        list_embeddings.append(n)

    # Output embedding choice
    if output_embedding == "graph":
        out = [PoolingNodes()([count_nodes, x, batch_id_node]) for x in list_embeddings]  # will return tensor
        out = [MLP(**last_mlp)(x) for x in out]
        out = [ks.layers.Dropout(dropout)(x) for x in out]
        out = ks.layers.Add()(out)
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":  # Node labeling
        out = GraphMLP(**last_mlp)([n, batch_id_node, count_nodes])
        out = GraphMLP(**output_mlp)([out, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `GIN` .")

    return out


def model_disjoint_edge(
        inputs,
        use_node_embedding: bool = None,
        use_edge_embedding: bool = None,
        input_node_embedding: dict = None,
        input_edge_embedding: dict = None,
        depth: int = None,
        gin_args: dict = None,
        gin_mlp: dict = None,
        last_mlp: dict = None,
        dropout: float = None,
        output_embedding: str = None,
        output_mlp: dict = None):

    n, ed, disjoint_indices, batch_id_node, count_nodes = inputs

    # Embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)
    if use_edge_embedding:
        ed = Embedding(**input_edge_embedding)(ed)

    # Model
    # Map to the required number of units.
    n_units = gin_mlp["units"][-1] if isinstance(gin_mlp["units"], list) else int(gin_mlp["units"])
    n = Dense(n_units, use_bias=True, activation='linear')(n)
    ed = Dense(n_units, use_bias=True, activation='linear')(ed)
    list_embeddings = [n]
    for i in range(0, depth):
        n = GINE(**gin_args)([n, disjoint_indices, ed])
        n = GraphMLP(**gin_mlp)([n, batch_id_node, count_nodes])
        list_embeddings.append(n)

    # Output embedding choice
    if output_embedding == "graph":
        out = [PoolingNodes()([count_nodes, x, batch_id_node]) for x in list_embeddings]  # will return tensor
        out = [MLP(**last_mlp)(x) for x in out]
        out = [ks.layers.Dropout(dropout)(x) for x in out]
        out = ks.layers.Add()(out)
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":  # Node labeling
        out = GraphMLP(**last_mlp)([n, batch_id_node, count_nodes])
        out = GraphMLP(**output_mlp)([out, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for mode `GINE` .")

    return out
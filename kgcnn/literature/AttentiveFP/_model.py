import keras as ks
from kgcnn.ops.activ import *
from kgcnn.layers.attention import AttentiveHeadFP
from kgcnn.layers.mlp import MLP, GraphMLP
from kgcnn.layers.modules import Embedding
from kgcnn.layers.update import GRUUpdate
from kgcnn.layers.pooling import PoolingNodesAttentive


def model_disjoint(
        inputs,
        use_node_embedding: bool = None,
        use_edge_embedding: bool = None,
        input_node_embedding: dict = None,
        input_edge_embedding: dict = None,
        depthmol: int = None,
        depthato: int = None,
        dropout: float = None,
        attention_args: dict = None,
        output_embedding: str = None,
        output_mlp: dict = None
):
    n, ed, edi, batch_id_node, count_nodes = inputs

    # Embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)
    if use_edge_embedding:
        ed = Embedding(**input_edge_embedding)(ed)

    # Model
    nk = ks.layers.Dense(units=attention_args['units'])(n)
    ck = AttentiveHeadFP(use_edge_features=True, **attention_args)([nk, ed, edi])
    nk = GRUUpdate(units=attention_args['units'])([nk, ck])

    for i in range(1, depthato):
        ck = AttentiveHeadFP(**attention_args)([nk, ed, edi])
        nk = GRUUpdate(units=attention_args['units'])([nk, ck])
        nk = ks.layers.Dropout(rate=dropout)(nk)
    n = nk

    # Output embedding choice
    if output_embedding == 'graph':
        out = PoolingNodesAttentive(units=attention_args['units'], depth=depthmol)([count_nodes, n, batch_id_node])
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported graph embedding for mode `AttentiveFP`")

    return out
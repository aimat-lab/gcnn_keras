import keras as ks
from keras.layers import Concatenate, Dense, Add, Activation, Dropout
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing, GatherState
from kgcnn.layers.mlp import MLP, GraphMLP
from kgcnn.layers.modules import Embedding
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.literature.DMPNN._layers import DMPNNPPoolingEdgesDirected


def model_disjoint(inputs: list = None,
                   use_node_embedding: bool = None,
                   use_edge_embedding: bool = None,
                   use_graph_embedding: bool = None,
                   input_node_embedding: dict = None,
                   input_edge_embedding: dict = None,
                   input_graph_embedding: dict = None,
                   pooling_args: dict = None,
                   edge_initialize: dict = None,
                   edge_dense: dict = None,
                   edge_activation: dict = None,
                   node_dense: dict = None,
                   dropout: dict = None,
                   depth: int = None,
                   use_graph_state: bool = False,
                   output_embedding: str = None,
                   output_mlp: dict = None):
    n, ed, edi, batch_id_node, ed_pairs, count_nodes, graph_state = inputs

    # Embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)
    if use_edge_embedding:
        ed = Embedding(**input_edge_embedding)(ed)
    if use_graph_state:
        if use_graph_embedding:
            graph_state = Embedding(**input_graph_embedding)(graph_state)

    # Make first edge hidden h0
    h_n0 = GatherNodesOutgoing()([n, edi])
    h0 = Concatenate(axis=-1)([h_n0, ed])
    h0 = Dense(**edge_initialize)(h0)

    # One Dense layer for all message steps
    edge_dense_all = Dense(**edge_dense)  # Should be linear activation

    # Model Loop
    h = h0
    for i in range(depth):
        m_vw = DMPNNPPoolingEdgesDirected()([n, h, edi, ed_pairs])
        h = edge_dense_all(m_vw)
        h = Add()([h, h0])
        h = Activation(**edge_activation)(h)
        if dropout is not None:
            h = Dropout(**dropout)(h)

    mv = AggregateLocalEdges(**pooling_args)([n, h, edi])
    mv = Concatenate(axis=-1)([mv, n])
    hv = Dense(**node_dense)(mv)

    # Output embedding choice
    n = hv
    if output_embedding == 'graph':
        out = PoolingNodes(**pooling_args)([count_nodes, n, batch_id_node])
        if use_graph_state:
            out = ks.layers.Concatenate()([graph_state, out])
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        if use_graph_state:
            graph_state_node = GatherState()([graph_state, n])
            n = Concatenate()([n, graph_state_node])
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported embedding mode for `DMPNN`.")

    return out

from keras.layers import Dense, Add, Multiply, Activation, Dropout, Subtract, Concatenate
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing, GatherEdgesPairs
from kgcnn.layers.modules import Embedding
from kgcnn.layers.mlp import GraphMLP, MLP
from ._layers import PoolingNodesGRU


def model_disjoint(
        inputs,
        use_node_embedding=None,
        use_edge_embedding=None,
        input_node_embedding=None,
        input_edge_embedding=None,
        node_initialize=None,
        edge_initialize=None,
        depth=None,
        pooling_kwargs=None,
        edge_dense=None,
        edge_activation=None,
        dropout=None,
        node_dense=None,
        output_embedding=None,
        use_final_gru=None,
        output_mlp=None,
        pooling_gru=None
):
    n, ed, edi, e_pairs, batch_id_node, node_id, count_nodes = inputs
    # Embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)
    if use_edge_embedding:
        ed = Embedding(**input_edge_embedding)(ed)

    h0 = Dense(**node_initialize)(n)
    he0 = Dense(**edge_initialize)(ed)

    # Model Loop
    h = h0
    he = he0
    for i in range(depth - 1):
        # Node message/update
        m_pool = AggregateLocalEdges(**pooling_kwargs)([h, he, edi])
        m_max = AggregateLocalEdges(pooling_method="max")([h, he, edi])
        m = Multiply()([m_pool, m_max])
        # In paper there is a potential COMMUNICATE() here but in reference code just add() operation.
        # Alternatives were GRU or MLP on concatenated messages.
        h = Add()([h, m])

        # Edge message/update
        h_out = GatherNodesOutgoing()([h, edi])
        e_rev = GatherEdgesPairs()([he, e_pairs])
        he = Subtract()([h_out, e_rev])
        he = Dense(**edge_dense)(he)
        he = Add()([he, he0])
        he = Activation(**edge_activation)(he)
        if dropout:
            he = Dropout(**dropout)(he)

    # Last step
    m_pool = AggregateLocalEdges(**pooling_kwargs)([h, he, edi])
    m_max = AggregateLocalEdges(pooling_method="max")([h, he, edi])
    m = Multiply()([m_pool, m_max])
    h_final = Concatenate()([m, h, h0])
    h_final = Dense(**node_dense)(h_final)

    n = h_final
    if output_embedding == 'graph':
        if use_final_gru:
            # Actually a simple GRU is not permutation invariant.
            # Better replace this by e.g. set2set or AttentivePooling.
            out = PoolingNodesGRU(**pooling_gru)([n, batch_id_node, node_id, count_nodes])
        else:
            out = PoolingNodes(**pooling_kwargs)([count_nodes, n, batch_id_node])
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported graph embedding for mode `CMPNN`")

    return out

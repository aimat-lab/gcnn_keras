from kgcnn.layers.modules import Embedding
from kgcnn.layers.mlp import MLP, GraphMLP
from kgcnn.layers.gather import GatherNodesOutgoing
from keras.layers import Concatenate, Dense, Activation, Add, Dropout
from kgcnn.layers.gather import GatherState
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.pooling import PoolingNodes
from ._layers import DMPNNPPoolingEdgesDirected, GIN_D


def model_disjoint(
        inputs,
        use_node_embedding,
        use_edge_embedding,
        use_graph_embedding,
        use_graph_state=None,
        input_node_embedding=None,
        input_edge_embedding=None,
        input_graph_embedding=None,
        edge_initialize=None,
        edge_activation=None,
        edge_dense=None,
        depthDMPNN=None,
        dropoutDMPNN=None,
        pooling_args=None,
        gin_mlp=None,
        depthGIN=None,
        gin_args=None,
        output_embedding=None,
        node_pooling_kwargs=None,
        last_mlp=None,
        dropoutGIN=None,
        output_mlp=None
):
    n, ed, edi, batch_id_node, ed_pairs, count_nodes, graph_state = inputs

    # Embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)
    if use_edge_embedding:
        ed = Embedding(**input_edge_embedding)(ed)
    if use_graph_state:
        if use_graph_embedding:
            graph_state = Embedding(**input_graph_embedding)(graph_state)

    # Make first edge hidden h0 step 1
    h_n0 = GatherNodesOutgoing()([n, edi])
    h0 = Concatenate(axis=-1)([h_n0, ed])
    h0 = Dense(**edge_initialize)(h0)
    h0 = Activation(**edge_activation)(h0)  # relu equation 1

    # One Dense layer for all message steps this is not the case in DGIN they are independents!
    edge_dense_all = Dense(**edge_dense)  # see equation 3 comments

    # Model Loop steps 2 & 3
    h = h0
    for i in range(depthDMPNN):
        # equation 2
        m_vw = DMPNNPPoolingEdgesDirected()([n, h, edi, ed_pairs])  # ed_pairs for Directed Pooling!
        # equation 3
        h = edge_dense_all(m_vw)  # do one per layer ...
        # h = Dense(**edge_dense)(m_vw)
        h = Add()([h, h0])
        # remark : dropout before Activation in DGIN code
        h = Activation(**edge_activation)(h)
        if dropoutDMPNN is not None:
            h = Dropout(**dropoutDMPNN)(h)

    # equation 4 & 5
    m_v = AggregateLocalEdges(**pooling_args)([n, h, edi])
    m_v = Concatenate(axis=-1)([n, m_v])  #
    # equation 5b: hv = Dense(**node_dense)(mv) removed based on the paper

    # GIN_D part (this projection is normally not done in DGIN, but we need to get the correct "dim")
    n_units = gin_mlp["units"][-1] if isinstance(gin_mlp["units"], list) else int(gin_mlp["units"])
    h_v = Dense(n_units, use_bias=True, activation='linear')(m_v)
    h_v_0 = h_v

    list_embeddings = [h_v_0]  # empty in the paper
    for i in range(depthGIN):
        # not sure of the mv, mv ... here but why not ;-)
        h_v = GIN_D(**gin_args)(
            [h_v, edi, h_v_0])  # equation 6 & 7a  mv is new the new nodes values and we do pooling on ed via edi
        h_v = GraphMLP(**gin_mlp)([h_v, batch_id_node, count_nodes])  # equation 7b
        list_embeddings.append(h_v)

    # Output embedding choice look like it takes only the last h_v in the paper not all ???
    if output_embedding == 'graph':
        out = [
            PoolingNodes(**node_pooling_kwargs)([count_nodes, x, batch_id_node]) for x in list_embeddings
        ]  # will return tensor equation 8
        out = [MLP(**last_mlp)(x) for x in out]  # MLP apply per depthGIN
        if dropoutGIN is not None:
            out = [Dropout(**dropoutGIN)(x) for x in out]
        out = Add()(out)
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        if use_graph_state:
            graph_state_node = GatherState()([graph_state, batch_id_node])
            n = Concatenate()([n, graph_state_node])
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported graph embedding for mode `DGIN` .")

    return out

from keras.layers import Flatten, Dense
from kgcnn.layers.mlp import MLP, GraphMLP
from kgcnn.layers.modules import Embedding, ZerosLike
from ._layers import HamNaiveDynMessage, HamNetFingerprintGenerator, HamNetNaiveUnion, HamNetGRUUnion


def model_disjoint(
        inputs,
        use_node_embedding,
        use_edge_embedding,
        input_node_embedding=None,
        input_edge_embedding=None,
        given_coordinates=None,
        gru_kwargs=None,
        message_kwargs=None,
        fingerprint_kwargs=None,
        output_embedding=None,
        output_mlp=None,
        union_type_edge=None,
        union_type_node=None,
        depth=None
):
    # Model implementation with disjoint representation.
    n, q_ftr, ed, edi, batch_id_node, count_nodes = inputs

    # Embedding, if no feature dimension
    if use_node_embedding:
        n = Embedding(**input_node_embedding)(n)
    if use_edge_embedding:
        ed = Embedding(**input_edge_embedding)(ed)

    # Generate coordinates.
    if given_coordinates:
        p_ftr = ZerosLike()(q_ftr)
    else:
        # Use Hamiltonian engine to get p, q coordinates.
        raise NotImplementedError("Hamiltonian engine not yet implemented")

    # Initialization
    n = Dense(units=gru_kwargs["units"], activation="tanh")(n)
    ed = Dense(units=gru_kwargs["units"], activation="tanh")(ed)
    p = p_ftr
    q = q_ftr

    # Message passing.
    for i in range(depth):
        # Message step
        nu, eu = HamNaiveDynMessage(**message_kwargs)([n, ed, p, q, edi])

        # Node updates
        if union_type_node == "gru":
            n = HamNetGRUUnion(**gru_kwargs)([n, nu])
        elif union_type_node == "naive":
            n = HamNetNaiveUnion(units=gru_kwargs["units"])([n, nu])
        else:
            n = nu

        # Edge updates
        if union_type_edge == "gru":
            ed = HamNetGRUUnion(**gru_kwargs)([ed, eu])
        elif union_type_edge == "naive":
            ed = HamNetNaiveUnion(units=gru_kwargs["units"])([ed, eu])
        else:
            ed = eu

    # Fingerprint generator for graph embedding.
    if output_embedding == 'graph':
        out = HamNetFingerprintGenerator(**fingerprint_kwargs)([count_nodes, n, batch_id_node])
        out = Flatten()(out)  # will be tensor.
        out = MLP(**output_mlp)(out)
    elif output_embedding == 'node':
        out = GraphMLP(**output_mlp)([n, batch_id_node, count_nodes])
    else:
        raise ValueError("Unsupported output embedding for `HamNet` .")

    return out

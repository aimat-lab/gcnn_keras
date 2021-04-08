import tensorflow.keras as ks


def generate_standard_graph_input(input_node_shape,
                                 input_edge_shape,
                                 input_state_shape,
                                 input_node_vocab=95,
                                 input_edge_vocab=5,
                                 input_state_vocab=100,
                                 input_node_embedd=64,
                                 input_edge_embedd=64,
                                 input_state_embedd=64,
                                 input_type='ragged'):
    """Generate input for a standard graph format.
    This includes nodes, edge, edge_indices and optional a graph state.

    Args:
        input_node_shape:
        input_edge_shape:
        input_state_shape:
        input_node_vocab:
        input_edge_vocab:
        input_state_vocab:
        input_node_embedd:
        input_edge_embedd:
        input_state_embedd:
        input_type:

    Returns:
        list: [node_input, node_embedding, edge_input, edge_embedding, edge_index_input, state_input, state_embedding]
    """
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    edge_input = ks.layers.Input(shape=input_edge_shape, name='edge_input', dtype="float32", ragged=True)
    edge_index_input = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)

    if len(input_node_shape) == 1:
        n = ks.layers.Embedding(input_node_vocab, input_node_embedd, name='node_embedding')(node_input)
    else:
        n = node_input

    if len(input_edge_shape) == 1:
        ed = ks.layers.Embedding(input_edge_vocab, input_edge_embedd, name='edge_embedding')(edge_input)
    else:
        ed = edge_input

    if input_state_shape is not None:
        env_input = ks.Input(shape=input_state_shape, dtype='float32', name='state_input')
        if len(input_state_shape) == 0:
            uenv = ks.layers.Embedding(input_state_vocab, input_state_embedd, name='state_embedding')(env_input)
        else:
            uenv = env_input

    if input_state_shape is not None:
        return node_input, n, edge_input, ed, edge_index_input,  env_input,   uenv,
    else:
        return node_input,  n, edge_input, ed, edge_index_input, None, None



def update_model_args(default_args = None, user_args = None):
    """
    Make arg dict with updated default values.

    Args:
        default_args:
        user_args:

    Returns:

    """
    out = {}
    if default_args is None:
        default_args = {}
    if user_args is None:
        user_args = {}
    out.update(default_args)
    out.update(user_args)
    return out
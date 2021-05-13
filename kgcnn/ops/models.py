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
                                  input_tensor_type='ragged'):
    """Generate input for a standard graph tensor format.
    This includes nodes, edge, edge_indices and optional a graph state.
    If input shape is (None,) a embedding layer is used to make the feature dimension.

    Args:
        input_node_shape (list): Shape of node input without batch dimension. Either (None,F) or (None,)
        input_edge_shape (list): Shape of edge input without batch dimension. Either (None,F) or (None,)
        input_state_shape: Shape of state input without batch dimension. Either (F,) or (,)
        input_node_vocab (int): Vocabulary size of optional embedding layer.
        input_edge_vocab (int): Vocabulary size of optional embedding layer.
        input_state_vocab (int) Vocabulary size of optional embedding layer.
        input_node_embedd (int): Embedding dimension for optional embedding layer.
        input_edge_embedd (int): Embedding dimension for optional embedding layer.
        input_state_embedd (int): Embedding dimension for optional embedding layer.
        input_tensor_type (str): Type of input tensor. Only "ragged" is supported at the moment.

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
        return node_input, n, edge_input, ed, edge_index_input, env_input, uenv,
    else:
        return node_input, n, edge_input, ed, edge_index_input, None, None


def update_model_args(default_args=None, user_args=None):
    """
    Make arg dict with updated default values.

    Args:
        default_args (dict): Dictionary of default values.
        user_args (dict): Dictionary of args from.

    Returns:
        dict: Make new dict and update with first default and then user args.
    """
    out = {}
    if default_args is None:
        default_args = {}
    if user_args is None:
        user_args = {}
    out.update(default_args)
    out.update(user_args)
    return out


def generate_mol_graph_input(input_node_shape,
                             input_xyz_shape,
                             input_bond_index_shape=None,
                             input_angle_index_shape=None,
                             input_dihedral_index_shape=None,
                             input_node_vocab=95,
                             input_node_embedd=64,
                             input_tensor_type='ragged'):
    """Generate input for a standard graph tensor format.
    This includes nodes, edge, edge_indices and optional a graph state.
    If input shape is (None,) a embedding layer is used to make the feature dimension.

    Args:
        input_node_shape (list): Shape of node input without batch dimension. Either (None,F) or (None,)
        input_xyz_shape (list): Shape of xyz input without batch dimension (None,3).
        input_node_vocab (int): Vocabulary size of optional embedding layer.
        input_node_embedd (int): Embedding dimension for optional embedding layer.
        input_tensor_type (str): Type of input tensor. Only "ragged" is supported at the moment.

    Returns:
        list: [node_input, node_embedding, edge_input, edge_embedding, edge_index_input, state_input, state_embedding]
    """
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    xyz_input = ks.layers.Input(shape=input_xyz_shape, name='xyz_input', dtype="float32", ragged=True)

    if input_bond_index_shape is not None:
        bond_index_input = ks.layers.Input(shape=input_bond_index_shape, name='bond_index_input', dtype="int64",
                                           ragged=True)
    else:
        bond_index_input = None

    if input_angle_index_shape is not None:
        angle_index_input = ks.layers.Input(shape=input_angle_index_shape, name='angle_index_input', dtype="int64",
                                            ragged=True)
    else:
        angle_index_input = None

    if input_dihedral_index_shape is not None:
        dihedral_index_input = ks.layers.Input(shape=input_dihedral_index_shape, name='dihedral_index_input', dtype="int64",
                                               ragged=True)
    else:
        dihedral_index_input = None

    if len(input_node_shape) == 1:
        n = ks.layers.Embedding(input_node_vocab, input_node_embedd, name='node_embedding')(node_input)
    else:
        n = node_input

    return node_input, n, xyz_input, bond_index_input, angle_index_input, dihedral_index_input

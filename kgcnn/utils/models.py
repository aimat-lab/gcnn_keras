import tensorflow.keras as ks


def generate_node_embedding(node_input, input_node_shape, embedding_args, **kwargs):
    """Optional node embedding for tensor input.

    Args:
        node_input (tf.Tensor): Input tensor to make embedding for.
        input_node_shape (list): Shape of node input without batch dimension. Either (None, F) or (None, )
        embedding_args (dict): Arguments for embedding layer.

    Returns:
        tf.Tensor: Tensor output.
    """
    if len(input_node_shape) == 1:
        n = ks.layers.Embedding(**embedding_args)(node_input)
    else:
        n = node_input
    return n


def generate_edge_embedding(edge_input, input_edge_shape, embedding_args, **kwargs):
    """Optional edge embedding for tensor input.

    Args:
        edge_input (tf.Tensor): Input tensor to make embedding for.
        input_edge_shape (list): Shape of edge input without batch dimension. Either (None, F) or (None, )
        embedding_args (dict): Arguments for embedding layer.

    Returns:
        tf.Tensor: Tensor output.
    """
    if len(input_edge_shape) == 1:
        ed = ks.layers.Embedding(**embedding_args)(edge_input)
    else:
        ed = edge_input
    return ed


def generate_state_embedding(env_input, input_state_shape, embedding_args, **kwargs):
    """Optional state embedding for tensor input.

    Args:
        env_input (tf.Tensor): Input tensor to make embedding for.
        input_state_shape: Shape of state input without batch dimension. Either (F, ) or (, )
        embedding_args (dict): Arguments for embedding layer.

    Returns:
        tf.Tensor: Tensor output.
    """
    if len(input_state_shape) == 0:
        uenv = ks.layers.Embedding(**embedding_args)(env_input)
    else:
        uenv = env_input
    return uenv


def generate_standard_graph_input(input_node_shape,
                                  input_edge_shape,
                                  input_state_shape,
                                  embedding_nodes,
                                  embedding_edges,
                                  embedding_state,
                                  **kwargs):
    """Generate input for a standard graph tensor format.
    This includes nodes, edge, edge_indices and optional a graph state.
    If input shape is (None, ) a embedding layer is used to make the feature dimension.

    Args:
        input_node_shape (list): Shape of node input without batch dimension. Either (None, F) or (None, )
        input_edge_shape (list): Shape of edge input without batch dimension. Either (None, F) or (None, )
        input_state_shape: Shape of state input without batch dimension. Either (F, ) or (, )
        embedding_nodes (dict): Arguments for embedding layer.
        embedding_edges (dict): Arguments for embedding layer.
        embedding_state (dict): Arguments for embedding layer.

    Returns:
        list: [node_input, node_embedding, edge_input, edge_embedding, edge_index_input, state_input, state_embedding]
    """
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    edge_input = ks.layers.Input(shape=input_edge_shape, name='edge_input', dtype="float32", ragged=True)
    edge_index_input = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)

    if len(input_node_shape) == 1:
        n = ks.layers.Embedding(**embedding_nodes)(node_input)
    else:
        n = node_input

    if len(input_edge_shape) == 1:
        ed = ks.layers.Embedding(**embedding_edges)(edge_input)
    else:
        ed = edge_input

    env_input = None
    uenv = None
    if input_state_shape is not None:
        env_input = ks.Input(shape=input_state_shape, dtype='float32', name='state_input')
        if len(input_state_shape) == 0:
            uenv = ks.layers.Embedding(**embedding_state)(env_input)
        else:
            uenv = env_input

    return node_input, n, edge_input, ed, edge_index_input, env_input, uenv


def update_model_args(default_args=None, user_args=None):
    """Make model parameter dictionary with updated default values.

    Args:
        default_args (dict): Dictionary of default values.
        user_args (dict): Dictionary of args to update.

    Returns:
        dict: Make new dict and update with first default and then user args.
    """
    out = {}
    if default_args is None:
        default_args = {}
    if user_args is None:
        user_args = {}

    # Check valid args
    for iter_key in user_args.keys():
        if iter_key not in default_args:
            raise ValueError("Model arg", iter_key, "not in default arguments", default_args.keys())

    out.update(default_args)

    # Nested update of args:
    def _nested_update(dict1, dict2):
        for key, values in dict2.items():
            if key not in dict1:
                print("WARNING: Unknown model argument:", key, "with value", values)
                dict1[key] = values
            else:
                if isinstance(dict1[key], dict) and isinstance(values, dict):
                    # The value is a dict of model arguments itself. Update the same way.
                    dict1[key] = _nested_update(dict1[key], values)
                elif isinstance(dict1[key], dict) and not isinstance(values, dict):
                    # If values is None, means no information, keep dict1 values untouched.
                    if values is not None:
                        raise ValueError("Error: Can not overwriting dictionary of", key, "with", values)
                else:
                    # Just any other value to update
                    dict1[key] = values
        return dict1

    return _nested_update(out, user_args)


def generate_mol_graph_input(input_node_shape,
                             input_xyz_shape,
                             input_bond_index_shape=None,
                             input_angle_index_shape=None,
                             input_dihedral_index_shape=None,
                             embedding_args=None,
                             **kwargs):
    """Generate input for a standard mol-graph tensor format.
    This includes nodes, coordinates, edge_indices and optional angle and dihedral indices.
    If input shape is (None, ) a embedding layer is used to make the feature dimension.

    Args:
        input_node_shape (list): Shape of node input without batch dimension. Either (None, F) or (None, )
        input_xyz_shape (list): Shape of xyz input without batch dimension (None, 3).
        input_bond_index_shape (list): Shape of the bond indices. Not used if set to None.
        input_angle_index_shape (list): Shape of the angle indices. Not used if set to None.
        input_dihedral_index_shape (list): Shape of the dihedral indices. Not used if set to None.
        embedding_args (dict): Arguments for embedding layer

    Returns:
        list: [node_input, node_embedding, coordinates, edge_index, angle_index, dihedral_index]
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
        dihedral_index_input = ks.layers.Input(shape=input_dihedral_index_shape, name='dihedral_index_input',
                                               dtype="int64", ragged=True)
    else:
        dihedral_index_input = None

    if len(input_node_shape) == 1:
        n = ks.layers.Embedding(**embedding_args)(node_input)
    else:
        n = node_input

    return node_input, n, xyz_input, bond_index_input, angle_index_input, dihedral_index_input

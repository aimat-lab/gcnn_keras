import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.gather import GatherState, GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.layers.keras import Concatenate, Dense
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingNodes
from kgcnn.layers.set2set import Set2Set
from kgcnn.ops.models import generate_node_embedding, update_model_args, generate_state_embedding, \
    generate_edge_embedding


# import tensorflow as tf


# 'Interaction Networks for Learning about Objects,Relations and Physics'
# by Peter W. Battaglia, Razvan Pascanu, Matthew Lai, Danilo Rezende, Koray Kavukcuoglu
# http://papers.nips.cc/paper/6417-interaction-networks-for-learning-about-objects-relations-and-physics
# https://github.com/higgsfield/interaction_network_pytorch


def make_inorp(  # Input
        input_node_shape,
        input_edge_shape,
        input_state_shape,
        input_embedding: dict = None,
        # Output
        output_embedding: dict = None,
        output_mlp: dict = None,
        # Model specific parameter
        depth=3,
        use_set2set: bool = False,  # not in original paper
        node_mlp_args: dict = None,
        edge_mlp_args: dict = None,
        set2set_args: dict = None,
        pooling_args: dict = None
):
    """Generate Interaction network.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_state_shape (list): Shape of state features. If shape is (,) embedding layer is used.
        input_embedding (dict): Dictionary of embedding parameters used if input shape is None. Default is
            {"nodes": {"input_dim": 95, "output_dim": 64},
            "edges": {"input_dim": 5, "output_dim": 64},
            "state": {"input_dim": 100, "output_dim": 64}}.
        output_embedding (dict): Dictionary of embedding parameters of the graph network. Default is
            {"output_mode": 'graph', "output_tensor_type": 'padded'}.
        output_mlp (dict): Dictionary of arguments for final MLP regression or classification layer. Default is
            {"use_bias": [True, True, False], "units": [25, 10, 1],
            "activation": ['relu', 'relu', 'sigmoid']}.
        depth (int): Number of convolution layers. Default is 3.
        use_set2set (str): Use set2set pooling for graph embedding. Default is False.
        node_mlp_args (dict): Dictionary of arguments for MLP for node update. Default is
            {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]}
        edge_mlp_args (dict): Dictionary of arguments for MLP for interaction update. Default is
            {"units": [100, 100, 100, 100, 50],
            "activation": ['relu', 'relu', 'relu', 'relu', "linear"]}
        set2set_args (dict): Dictionary of set2set layer arguments. Default is
            {'channels': 32, 'T': 3, "pooling_method": "mean", "init_qstar": "mean"}.
        pooling_args (dict): Dictionary for message pooling arguments. Default is
            {'pooling_method': "segment_mean"}

    Returns:
        tf.keras.models.Model: Interaction model.
    """
    # default values
    model_default = {'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 64},
                                         "edges": {"input_dim": 5, "output_dim": 64},
                                         "state": {"input_dim": 100, "output_dim": 64}},
                     'output_embedding': {"output_mode": 'graph', "output_tensor_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'sigmoid']},
                     'set2set_args': {'channels': 32, 'T': 3, "pooling_method": "mean",
                                      "init_qstar": "mean"},
                     'node_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
                     'edge_mlp_args': {"units": [100, 100, 100, 100, 50],
                                       "activation": ['relu', 'relu', 'relu', 'relu', "linear"]},
                     'pooling_args': {'pooling_method': "segment_mean"}
                     }

    # Update default values
    input_embedding = update_model_args(model_default['input_embedding'], input_embedding)
    output_embedding = update_model_args(model_default['output_embedding'], output_embedding)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    set2set_args = update_model_args(model_default['set2set_args'], set2set_args)
    node_mlp_args = update_model_args(model_default['node_mlp_args'], node_mlp_args)
    edge_mlp_args = update_model_args(model_default['edge_mlp_args'], edge_mlp_args)
    pooling_args = update_model_args(model_default['pooling_args'], pooling_args)
    gather_args = {"node_indexing": "sample"}

    # Make input embedding, if no feature dimension
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    edge_input = ks.layers.Input(shape=input_edge_shape, name='edge_input', dtype="float32", ragged=True)
    edge_index_input = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)
    env_input = ks.Input(shape=input_state_shape, dtype='float32', name='state_input')
    n = generate_node_embedding(node_input, input_node_shape, input_embedding['nodes'])
    ed = generate_edge_embedding(edge_input, input_edge_shape, input_embedding['edges'])
    uenv = generate_state_embedding(env_input, input_state_shape, input_embedding['state'])
    edi = edge_index_input

    # Preprocessing
    ev = GatherState(**gather_args)([uenv, n])
    # n-Layer Step
    for i in range(0, depth):
        # upd = GatherNodes()([n,edi])
        eu1 = GatherNodesIngoing(**gather_args)([n, edi])
        eu2 = GatherNodesOutgoing(**gather_args)([n, edi])
        upd = Concatenate(axis=-1)([eu2, eu1])
        eu = Concatenate(axis=-1)([upd, ed])

        eu = MLP(**edge_mlp_args)(eu)
        # Pool message
        nu = PoolingLocalEdges(**pooling_args)(
            [n, eu, edi])  # Summing for each node connection
        # Add environment
        nu = Concatenate(axis=-1)(
            [n, nu, ev])  # Concatenate node features with new edge updates

        n = MLP(**node_mlp_args)(nu)

    if output_embedding["output_mode"] == 'graph':
        if use_set2set:
            # output
            outss = Dense(set2set_args["channels"], activation="linear")(n)
            out = Set2Set(**set2set_args)(outss)
        else:
            out = PoolingNodes(**pooling_args)(n)
        main_output = MLP(**output_mlp)(out)

    else:  # Node labeling
        out = n
        main_output = MLP(**output_mlp)(out)

        main_output = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(main_output)
        # no ragged for distribution atm

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, env_input], outputs=main_output)

    return model

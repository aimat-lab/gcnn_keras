import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType, ChangeIndexing
from kgcnn.layers.gather import GatherState, GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.layers.keras import Concatenate, Dense
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingLocalEdges, PoolingNodes
from kgcnn.layers.set2set import Set2Set
from kgcnn.ops.models import generate_standard_graph_input, update_model_args


# import tensorflow as tf


# 'Interaction Networks for Learning about Objects,Relations and Physics'
# by Peter W. Battaglia, Razvan Pascanu, Matthew Lai, Danilo Rezende, Koray Kavukcuoglu
# http://papers.nips.cc/paper/6417-interaction-networks-for-learning-about-objects-relations-and-physics
# https://github.com/higgsfield/interaction_network_pytorch


def make_inorp(  # Input
        input_node_shape,
        input_edge_shape,
        input_state_shape,
        input_embedd: dict = None,
        # Output
        output_embedd: dict = None,
        output_mlp: dict = None,
        # Model specific parameter
        depth=3,
        use_set2set: bool = False,  # not in original paper
        node_mlp_args: dict = None,
        edge_mlp_args: dict = None,
        set2set_args: dict = None,
        pooling_args: dict = None
):
    """
    Generate Interaction network.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_state_shape (list): Shape of state features. If shape is (,) embedding layer is used.
        input_embedd (dict): Dictionary of embedding parameters used if input shape is None. Default is
            {'input_node_vocab': 95, 'input_edge_vocab': 5, 'input_state_vocab': 100,
            'input_node_embedd': 64, 'input_edge_embedd': 64, 'input_state_embedd': 64,
            'input_type': 'ragged'}.
        output_embedd (dict): Dictionary of embedding parameters of the graph network. Default is
            {"output_mode": 'graph', "output_type": 'padded'}.
        output_mlp (dict): Dictionary of arguments for final MLP regression or classifcation layer. Default is
            {"use_bias": [True, True, False], "units": [25, 10, 1],
            "activation": ['relu', 'relu', 'sigmoid']}.
        depth (int): Number of convolution layers. Default is 3.
        node_mlp_args (dict): Dictionary of arguments for MLP for node update. Default is
            {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]}
        edge_mlp_args (dict): Dictionary of arguments for MLP for interaction update. Default is
            {"units": [100, 100, 100, 100, 50],
            "activation": ['relu', 'relu', 'relu', 'relu', "linear"]}
        use_set2set (str): Use set2set pooling for graph embedding. Default is False.
        set2set_args (dict): Dictionary of set2set layer arguments. Default is
            {'channels': 32, 'T': 3, "pooling_method": "mean", "init_qstar": "mean"}.
        pooling_args (dict): Dictionary for message pooling arguments. Default is
            {'is_sorted': False, 'has_unconnected': True, 'pooling_method': "segment_mean"}

    Returns:
        model (tf.keras.model): Interaction model.

    """
    # default values
    model_default = {'input_embedd': {'input_node_vocab': 95, 'input_edge_vocab': 5, 'input_state_vocab': 100,
                                      'input_node_embedd': 64, 'input_edge_embedd': 64, 'input_state_embedd': 64,
                                      'input_tensor_type': 'ragged'},
                     'output_embedd': {"output_mode": 'graph', "output_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'sigmoid']},
                     'set2set_args': {'channels': 32, 'T': 3, "pooling_method": "mean",
                                      "init_qstar": "mean"},
                     'node_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
                     'edge_mlp_args': {"units": [100, 100, 100, 100, 50],
                                       "activation": ['relu', 'relu', 'relu', 'relu', "linear"]},
                     'pooling_args': {'is_sorted': False, 'has_unconnected': True, 'pooling_method': "segment_mean"}
                     }

    # Update default values
    input_embedd = update_model_args(model_default['input_embedd'], input_embedd)
    output_embedd = update_model_args(model_default['output_embedd'], output_embedd)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    set2set_args = update_model_args(model_default['set2set_args'], set2set_args)
    node_mlp_args = update_model_args(model_default['node_mlp_args'], node_mlp_args)
    edge_mlp_args = update_model_args(model_default['edge_mlp_args'], edge_mlp_args)
    pooling_args = update_model_args(model_default['pooling_args'], pooling_args)

    # Make input embedding, if no feature dimension
    node_input, n, edge_input, ed, edge_index_input, env_input, uenv = generate_standard_graph_input(input_node_shape,
                                                                                                     input_edge_shape,
                                                                                                     input_state_shape,
                                                                                                     **input_embedd)

    # Preprocessing
    tens_type = "values_partition"
    node_indexing = "batch"
    n = ChangeTensorType(input_tensor_type="ragged", output_tensor_type=tens_type)(n)
    ed = ChangeTensorType(input_tensor_type="ragged", output_tensor_type=tens_type)(ed)
    edi = ChangeTensorType(input_tensor_type="ragged", output_tensor_type=tens_type)(edge_index_input)
    edi = ChangeIndexing(input_tensor_type=tens_type, to_indexing=node_indexing)([n, edi])  # disjoint

    gather_args = {"input_tensor_type": tens_type, "node_indexing": node_indexing}
    edge_mlp_args.update({"input_tensor_type": tens_type})
    node_mlp_args.update({"input_tensor_type": tens_type})
    pooling_args.update({"input_tensor_type": tens_type, "node_indexing": node_indexing})
    set2set_args.update({"input_tensor_type": tens_type})

    ev = GatherState(**gather_args)([uenv, n])
    # n-Layer Step
    for i in range(0, depth):
        # upd = GatherNodes()([n,edi])
        eu1 = GatherNodesIngoing(**gather_args)([n, edi])
        eu2 = GatherNodesOutgoing(**gather_args)([n, edi])
        upd = Concatenate(axis=-1, input_tensor_type=tens_type)([eu2, eu1])
        eu = Concatenate(axis=-1, input_tensor_type=tens_type)([upd, ed])

        eu = MLP(**edge_mlp_args)(eu)
        # Pool message
        nu = PoolingLocalEdges(**pooling_args)(
            [n, eu, edi])  # Summing for each node connection
        # Add environment
        nu = Concatenate(axis=-1, input_tensor_type=tens_type)(
            [n, nu, ev])  # Concatenate node features with new edge updates

        n = MLP(**node_mlp_args)(nu)

    if output_embedd["output_mode"] == 'graph':
        if use_set2set:
            # output
            outss = Dense(set2set_args["channels"], input_tensor_type=tens_type, activation="linear")(n)
            out = Set2Set(**set2set_args)(outss)
        else:
            out = PoolingNodes(**pooling_args)(n)

        output_mlp.update({"input_tensor_type": "tensor"})
        main_output = MLP(**output_mlp)(out)

    else:  # Node labeling
        out = n
        main_output = MLP(**output_mlp)(out)

        main_output = ChangeTensorType(input_tensor_type=tens_type, output_tensor_type="tensor")(main_output)
        # no ragged for distribution atm

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, env_input], outputs=main_output)

    return model

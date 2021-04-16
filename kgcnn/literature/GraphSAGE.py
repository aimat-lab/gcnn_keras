from kgcnn.layers.disjoint.casting import CastRaggedToDisjoint, CastValuesToRagged
from kgcnn.layers.disjoint.mlp import MLP
from kgcnn.layers.disjoint.gather import GatherNodesOutgoing
from kgcnn.layers.disjoint.pooling import PoolingNodes,PoolingLocalMessages
from kgcnn.layers.ragged.casting import CastRaggedToDense
from kgcnn.utils.models import generate_standard_graph_input, update_model_args

import tensorflow.keras as ks
import tensorflow as tf

# 'Inductive Representation Learning on Large Graphs'
# William L. Hamilton and Rex Ying and Jure Leskovec
# http://arxiv.org/abs/1706.02216

def make_graph_sage(  # Input
        input_node_shape,
        input_edge_shape,
        input_embedd: dict = None,
        # Output
        output_embedd: dict = None,
        output_mlp: dict = None,
        # Model specific parameter
        depth=3,
        use_edge_features = False,
        node_mlp_args: dict = None,
        edge_mlp_args: dict = None,
        pooling_args: dict = None
):
    """
    Generate Interaction network.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
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
        use_edge_features (bool): Whether to concatenate edges with nodes in aggregate. Default is False.
        node_mlp_args (dict): Dictionary of arguments for MLP for node update. Default is
            {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]}
        edge_mlp_args (dict): Dictionary of arguments for MLP for interaction update. Default is
            {"units": [100, 100, 100, 100, 50],
            "activation": ['relu', 'relu', 'relu', 'relu', "linear"]}
        pooling_args (dict): Dictionary for message pooling arguments. Default is
            {'is_sorted': False, 'has_unconnected': True, 'pooling_method': "segment_mean"}

    Returns:
        model (tf.keras.model): Interaction model.

    """
    # default values
    model_default = {'input_embedd': {'input_node_vocab': 95, 'input_edge_vocab': 5, 'input_state_vocab': 100,
                                      'input_node_embedd': 64, 'input_edge_embedd': 64, 'input_state_embedd': 64,
                                      'input_type': 'ragged'},
                     'output_embedd': {"output_mode": 'graph', "output_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'sigmoid']},
                     'node_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
                     'edge_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu',  "linear"]},
                     'pooling_args': {'is_sorted': False, 'has_unconnected': True, 'pooling_method': "segment_mean"}
                     }

    # Update default values
    input_embedd = update_model_args(model_default['input_embedd'], input_embedd)
    output_embedd = update_model_args(model_default['output_embedd'], output_embedd)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    node_mlp_args = update_model_args(model_default['node_mlp_args'], node_mlp_args)
    edge_mlp_args = update_model_args(model_default['edge_mlp_args'], edge_mlp_args)
    pooling_args = update_model_args(model_default['pooling_args'], pooling_args)

    # Make input embedding, if no feature dimension
    node_input, n, edge_input, ed, edge_index_input, env_input, uenv = generate_standard_graph_input(input_node_shape,
                                                                                                     input_edge_shape,
                                                                                                     None,
                                                                                                     **input_embedd)

    # Make input embedding, if no feature dimension
    node_input, n, edge_input, ed, edge_index_input, _, _ = generate_standard_graph_input(input_node_shape,
                                                                                          input_edge_shape, None,
                                                                                          **input_embedd)

    n, node_len, ed, edge_len, edi = CastRaggedToDisjoint()([n, ed, edge_index_input])


    for i in range(0, depth):
        # upd = GatherNodes()([n,edi])
        eu = GatherNodesOutgoing()([n, node_len, edi, edge_len])
        if use_edge_features:
            eu = ks.layers.Concatenate(axis=-1)([eu, ed])

        eu = MLP(**edge_mlp_args)(eu)
        # Pool message
        nu = PoolingLocalMessages(**pooling_args)([n, node_len, eu, edge_len, edi])  # Summing for each node connection

        nu = ks.layers.Concatenate()([n, nu])  # Concatenate node features with new edge updates

        n = MLP(**node_mlp_args)(nu)
        n = tf.keras.layers.LayerNormalization(axis=-1)(n)  # Nomralize

    # Regression layer on output
    if output_embedd["output_mode"] == 'graph':
        out = PoolingNodes()([n, node_len])
        out = MLP(**output_mlp)(out)
        main_output = ks.layers.Flatten()(out)  # will be dense
    else:  # node embedding
        out = MLP(**output_mlp)(n)
        main_output = CastValuesToRagged()([out, node_len])
        main_output = CastRaggedToDense()(main_output)  # no ragged for distribution supported atm

    model = tf.keras.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)

    return model

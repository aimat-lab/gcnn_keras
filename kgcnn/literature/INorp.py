import tensorflow.keras as ks

from kgcnn.layers.disjoint.mlp import MLP
from kgcnn.layers.ragged.casting import CastRaggedToDense
from kgcnn.layers.ragged.conv import DenseRagged
from kgcnn.layers.ragged.gather import GatherState, GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.layers.ragged.mlp import MLPRagged
from kgcnn.layers.ragged.pooling import PoolingEdgesPerNode, PoolingNodes
# from kgcnn.layers.ragged.pooling import PoolingWeightedEdgesPerNode
from kgcnn.layers.ragged.set2set import Set2Set
from kgcnn.utils.models import generate_standard_graph_input, update_model_args


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
        is_sorted=True,
        has_unconnected=False,
        set2set_args: dict = None,
        pooling_method="segment_mean",
        **kwargs):
    """
    Generate Interaction network.
    
    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_state_shape (list): Shape of state features. If shape is (,) embedding layer is used.
        input_embedd (dict): Input embedding type.
        
        output_embedd (dict): Graph or node embedding of the graph network. Default is 'graph'.
        output_mlp (dict):
        
        is_sorted (bool, optional): Edge indices are sorted. Defaults to True.
        has_unconnected (bool, optional): Has unconnected nodes. Defaults to False.
        depth (int): Number of convolution layers. Default is 3.
        node_mlp_args (dict): Hidden parameter dfor multiple kernels. Default is [100,50].
        output_embedd (dict): Hidden parameter for multiple kernels. Default is [100,100,100,100,50].
        use_set2set (str): Use set2set pooling for graph embedding. Default is False.
        set2set_args (dict): Set2set dimension. Default is 32.
        pooling_method (str): Pooling method. Default is "segment_mean".
    
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
                     'set2set_args': {'channels': 32, 'T': 3, "pooling_method": "mean",
                                      "init_qstar": "mean"},
                     'node_mlp_args': {"units": [100, 50], "use_bias": True, "activation": ['relu', "linear"]},
                     'edge_mlp_args': {"units": [100, 100, 100, 100, 50],
                                       "activation": ['relu', 'relu', 'relu', 'relu', "linear"]}
                     }

    # Update default values
    input_embedd = update_model_args(model_default['input_embedd'], input_embedd)
    output_embedd = update_model_args(model_default['output_embedd'], output_embedd)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    set2set_args = update_model_args(model_default['set2set_args'], set2set_args)
    node_mlp_args = update_model_args(model_default['node_mlp_args'], node_mlp_args)
    edge_mlp_args = update_model_args(model_default['edge_mlp_args'], edge_mlp_args)

    # Make input embedding, if no feature dimension
    node_input, n, edge_input, ed, edge_index_input, env_input, uenv = generate_standard_graph_input(input_node_shape,
                                                                                                     input_edge_shape,
                                                                                                     input_state_shape,
                                                                                                     **input_embedd)

    # Preprocessing
    edi = edge_index_input
    # edi = ChangeIndexing()([n, edge_index_input])

    ev = GatherState()([uenv, n])
    # n-Layer Step
    for i in range(0, depth):
        # upd = GatherNodes()([n,edi])
        eu1 = GatherNodesIngoing()([n, edi])
        eu2 = GatherNodesOutgoing()([n, edi])
        upd = ks.layers.Concatenate(axis=-1)([eu2, eu1])
        eu = ks.layers.Concatenate(axis=-1)([upd, ed])

        eu = MLPRagged(**node_mlp_args)(eu)
        # Pool message
        nu = PoolingEdgesPerNode(pooling_method=pooling_method, is_sorted=is_sorted, has_unconnected=has_unconnected,
                                 )([n, eu, edi])  # Summing for each node connection
        # Add environment
        nu = ks.layers.Concatenate()([n, nu, ev])  # Concatenate node features with new edge updates

        n = MLPRagged(**edge_mlp_args)(nu)

    if output_embedd["output_mode"] == 'graph':
        if use_set2set:
            # output
            outss = DenseRagged(set2set_args["channels"])(n)
            out = Set2Set(**set2set_args)(outss)
        else:
            out = PoolingNodes()(n)

        main_output = MLP(**output_mlp)(out)

    else:  # Node labeling
        out = n
        main_output = MLPRagged(**output_mlp)(out)

        main_output = CastRaggedToDense()(main_output)  # no ragged for distribution supported atm

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, env_input], outputs=main_output)

    return model

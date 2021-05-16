import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType, ChangeIndexing
from kgcnn.layers.interaction import SchNetInteraction
from kgcnn.layers.keras import Dense
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.ops.models import generate_standard_graph_input, update_model_args


# Model Schnet as defined
# by Schuett et al. 2018
# https://doi.org/10.1063/1.5019779
# https://arxiv.org/abs/1706.08566  
# https://aip.scitation.org/doi/pdf/10.1063/1.5019779


def make_schnet(
        # Input
        input_node_shape,
        input_edge_shape,
        input_embedd: dict = None,
        # Output
        output_mlp: dict = None,
        output_dense: dict = None,
        output_embedd: dict = None,
        # Model specific
        depth=4,
        out_scale_pos=0,
        interaction_args: dict = None,
        node_pooling_args: dict = None
):
    """
    Make uncompiled SchNet model.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_embedd (list): Dictionary of embedding parameters used if input shape is None. Default is
            {'input_node_vocab': 95, 'input_edge_vocab': 5, 'input_state_vocab': 100,
            'input_node_embedd': 64, 'input_edge_embedd': 64, 'input_state_embedd': 64,
            'input_type': 'ragged'}
        output_mlp (dict, optional): Parameter for MLP output classification/ regression. Defaults to
            {"use_bias": [True, True], "units": [128, 64],
            "activation": ['shifted_softplus', 'shifted_softplus']}
        output_dense (dict): Parameter for Dense scaling layer. Defaults to {"units": 1, "activation": 'linear',
             "use_bias": True}.
        output_embedd (str): Dictionary of embedding parameters of the graph network. Default is
             {"output_mode": 'graph', "output_type": 'padded'}
        depth (int, optional): Number of Interaction units. Defaults to 4.
        out_scale_pos (int, optional): Scaling output, position of layer. Defaults to 0.
        interaction_args (dict): Interaction Layer arguments. Defaults include {"node_dim" : 128, "use_bias": True,
             "activation" : 'shifted_softplus', "cfconv_pool" : 'segment_sum',
             "is_sorted": False, "has_unconnected": True}
        node_pooling_args (dict, optional): Node pooling arguments. Defaults to {"pooling_method": "segment_sum"}.

    Returns:
        model (tf.keras.models.Model): SchNet.

    """
    # Make default values if None
    model_default = {'input_embedd': {'input_node_vocab': 95, 'input_edge_vocab': 5, 'input_state_vocab': 100,
                                      'input_node_embedd': 64, 'input_edge_embedd': 64, 'input_state_embedd': 64,
                                      'input_tensor_type': 'ragged'},
                     'output_embedd': {"output_mode": 'graph', "output_type": 'padded'},
                     'interaction_args': {"units": 128, "use_bias": True,
                                          "activation": 'shifted_softplus', "cfconv_pool": 'sum',
                                          "is_sorted": False, "has_unconnected": True},
                     'output_mlp': {"use_bias": [True, True], "units": [128, 64],
                                    "activation": ['shifted_softplus', 'shifted_softplus']},
                     'output_dense': {"units": 1, "activation": 'linear', "use_bias": True},
                     'node_pooling_args': {"pooling_method": "sum"}
                     }

    # Update args
    input_embedd = update_model_args(model_default['input_embedd'], input_embedd)
    interaction_args = update_model_args(model_default['interaction_args'], interaction_args)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    output_dense = update_model_args(model_default['output_dense'], output_dense)
    output_embedd = update_model_args(model_default['output_embedd'], output_embedd)
    node_pooling_args = update_model_args(model_default['node_pooling_args'], node_pooling_args)

    # Make input embedding, if no feature dimension
    node_input, n, edge_input, ed, edge_index_input, _, _ = generate_standard_graph_input(input_node_shape,
                                                                                          input_edge_shape, None,
                                                                                          **input_embedd)

    # Use representation
    tens_type = "ragged"
    node_indexing = "sample"
    partition_type = "row_length"
    n = ChangeTensorType(input_tensor_type="ragged", output_tensor_type=tens_type,partition_type=partition_type)(n)
    ed = ChangeTensorType(input_tensor_type="ragged", output_tensor_type=tens_type,partition_type=partition_type)(ed)
    edi = ChangeTensorType(input_tensor_type="ragged", output_tensor_type=tens_type,partition_type=partition_type)(edge_index_input)
    edi = ChangeIndexing(input_tensor_type=tens_type, to_indexing=node_indexing,partition_type=partition_type)([n, edi])

    n = Dense(interaction_args["units"], activation='linear',
              input_tensor_type=tens_type)(n)

    for i in range(0, depth):
        n = SchNetInteraction(input_tensor_type=tens_type, node_indexing=node_indexing,partition_type=partition_type,
                              **interaction_args)([n, ed, edi])

    n = MLP(input_tensor_type=tens_type,**output_mlp)(n)

    mlp_last = Dense(input_tensor_type=tens_type, **output_dense)

    if output_embedd["output_mode"] == 'graph':
        if out_scale_pos == 0:
            n = mlp_last(n)
        out = PoolingNodes(input_tensor_type=tens_type, node_indexing=node_indexing, partition_type=partition_type,
                           **node_pooling_args)(n)
        if out_scale_pos == 1:
            out = mlp_last(out)
        main_output = ks.layers.Flatten()(out)  # will be dense
    else:  # node embedding
        out = mlp_last(n)
        main_output = ChangeTensorType(input_tensor_type="values_partition", output_tensor_type="tensor",
                                       partition_type=partition_type)(out)  # no ragged for distribution atm

    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)

    return model

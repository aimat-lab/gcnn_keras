import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.attention import AttentiveHeadFP, AttentiveNodePooling
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.keras import Dense
from kgcnn.layers.update import GRUupdate
from kgcnn.layers.mlp import MLP
from kgcnn.ops.models import generate_standard_graph_input, update_model_args


# Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism
# Zhaoping Xiong, Dingyan Wang, Xiaohong Liu, Feisheng Zhong, Xiaozhe Wan, Xutong Li, Zhaojun Li, Xiaomin Luo, Kaixian Chen, Hualiang Jiang*, and Mingyue Zheng*
# Cite this: J. Med. Chem. 2020, 63, 16, 8749–8760
# Publication Date:August 13, 2019
# https://doi.org/10.1021/acs.jmedchem.9b00959


def make_attentiveFP(  # Input
        input_node_shape,
        input_edge_shape,
        input_embedd: dict = None,
        # Output
        output_embedd: dict = None,
        output_mlp: dict = None,
        # Model specific parameter
        depth=3,
        attention_args: dict = None
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
        attention_args (dict): Layer arguments for attention layer. Default is
            {"units": 32, 'is_sorted': False, 'has_unconnected': True}
    Returns:
        model (tf.keras.model): Interaction model.
    """

    # default values
    model_default = {'input_embedd': {'input_node_vocab': 95, 'input_edge_vocab': 5, 'input_state_vocab': 100,
                                      'input_node_embedd': 64, 'input_edge_embedd': 64, 'input_state_embedd': 64,
                                      'input_tensor_type': 'ragged'},
                     'output_embedd': {"output_mode": 'graph', "output_tensor_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'sigmoid']},
                     'attention_args': {"units": 32, 'is_sorted': False, 'has_unconnected': True}
                     }

    # Update default values
    input_embedd = update_model_args(model_default['input_embedd'], input_embedd)
    output_embedd = update_model_args(model_default['output_embedd'], output_embedd)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    attention_args = update_model_args(model_default['attention_args'], attention_args)

    # Make input embedding, if no feature dimension
    node_input, n, edge_input, ed, edge_index_input, _, _ = generate_standard_graph_input(input_node_shape,
                                                                                          input_edge_shape, None,
                                                                                          **input_embedd)



    edi = edge_index_input
    nk = Dense(units=attention_args['units'])(n)
    Ck = AttentiveHeadFP(use_edge_features=True,**attention_args)([nk,ed,edi])
    nk = GRUupdate(units=attention_args['units'])([nk,Ck])

    for i in range(1, depth):
        Ck = AttentiveHeadFP(**attention_args)([nk,ed,edi])
        nk = GRUupdate(units=attention_args['units'])([nk, Ck])

    n = nk
    if output_embedd["output_mode"] == 'graph':
        out = AttentiveNodePooling(units=attention_args['units'])(n)
        output_mlp.update({"input_tensor_type": "tensor"})
        out = MLP(**output_mlp)(out)
        main_output = ks.layers.Flatten()(out)  # will be dense
    else:  # node embedding
        out = MLP(**output_mlp)(n)
        main_output = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)

    model = tf.keras.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)

    return model

import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.layers.attention import AttentionHeadGATV2
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.keras import Concatenate, Dense, Average
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.ops.models import generate_node_embedding, update_model_args, generate_edge_embedding


# Graph Attention Networks by Veličković et al. (2018)
# https://arxiv.org/abs/1710.10903
# Improved by
# How Attentive are Graph Attention Networks?
# by Brody et al. (2021)

def make_gat_v2(  # Input
        input_node_shape,
        input_edge_shape,
        input_embedding: dict = None,
        # Output
        output_embedding: dict = None,
        output_mlp: dict = None,
        # Model specific parameter
        depth=3,
        attention_heads_num=5,
        attention_heads_concat=False,
        attention_args: dict = None
):
    """Generate Graph attention network.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
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
        attention_heads_num (int): Number of attention heads. Default is 5.
        attention_heads_concat (bool): Concat attention. Default is False.
        attention_args (dict): Layer arguments for attention layer. Default is
            {"units": 32, 'is_sorted': False}

    Returns:
        tf.keras.models.Model: GAT model.
    """
    # default values
    model_default = {'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 64},
                                         "edges": {"input_dim": 5, "output_dim": 64},
                                         "state": {"input_dim": 100, "output_dim": 64}},
                     'output_embedding': {"output_mode": 'graph', "output_tensor_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'sigmoid']},
                     'attention_args': {"units": 32, 'is_sorted': False}
                     }

    # Update default values
    input_embedding = update_model_args(model_default['input_embedding'], input_embedding)
    output_embedding = update_model_args(model_default['output_embedding'], output_embedding)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    attention_args = update_model_args(model_default['attention_args'], attention_args)
    pooling_nodes_args = {}

    # Make input embedding, if no feature dimension
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    edge_input = ks.layers.Input(shape=input_edge_shape, name='edge_input', dtype="float32", ragged=True)
    edge_index_input = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)
    n = generate_node_embedding(node_input, input_node_shape, input_embedding['nodes'])
    ed = generate_edge_embedding(edge_input, input_edge_shape, input_embedding['edges'])
    edi = edge_index_input

    nk = Dense(units=attention_args["units"], activation="linear")(n)
    for i in range(0, depth):
        heads = [AttentionHeadGATV2(**attention_args)([nk, ed, edi]) for _ in range(attention_heads_num)]
        if attention_heads_concat:
            nk = Concatenate(axis=-1)(heads)
        else:
            nk = Average()(heads)

    n = nk
    if output_embedding["output_mode"] == 'graph':
        out = PoolingNodes(**pooling_nodes_args)(n)
        out = MLP(**output_mlp)(out)
        main_output = ks.layers.Flatten()(out)  # will be dense
    else:  # node embedding
        out = MLP(**output_mlp)(n)
        main_output = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(out)

    model = tf.keras.models.Model(inputs=[node_input, edge_input, edge_index_input], outputs=main_output)

    return model

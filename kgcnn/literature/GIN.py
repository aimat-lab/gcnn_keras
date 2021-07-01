import tensorflow as tf
import tensorflow.keras as ks

from kgcnn.ops.models import update_model_args
from kgcnn.ops.models import generate_node_embedding
from kgcnn.layers.keras import Dense, Concatenate, Dropout, Add, BatchNormalization, Activation
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.mlp import MLP
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv import GIN


# How Powerful are Graph Neural Networks?
# Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka
# https://arxiv.org/abs/1810.00826


def make_gin(
        # Input
        input_node_shape,
        input_embedd: dict = None,
        # Output
        output_embedd: dict = None,
        output_mlp: dict = None,
        # Model specific
        depth=3,
        output_activation="softmax",
        dropout=0.0,
        gin_args: dict = None
):
    """Make GCN model.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_embedd (dict): Dictionary of embedding parameters used if input shape is None. Default is
            {"input_node_vocab": 100, "input_edge_vocab": 10, "input_state_vocab": 100,
            "input_node_embedd": 64, "input_edge_embedd": 64, "input_state_embedd": 64,
            "input_tensor_type": 'ragged'}.
        output_embedd (dict): Dictionary of embedding parameters of the graph network. Default is
            {"output_mode": 'graph', "output_tensor_type": 'padded'}.
        output_mlp (dict): Dictionary of arguments for final MLP regression or classification layer. Default is
            {"use_bias": [True, True, False], "units": [25, 10, 1],
            "activation": ['relu', 'relu', 'sigmoid']}.
        output_activation (str): Final activation at output. Defaults to "softmax".
        depth (int, optional): Number of convolutions. Defaults to 3.
        gin_args (dict): Dictionary of arguments for the GCN convolutional unit. Defaults to
            {"units": [64, 64], "use_bias": True, "activation": ['relu', 'linear'], "pooling_method": 'sum',
            "is_sorted": False, "has_unconnected": "True"}.

    Returns:
        tf.keras.models.Model: Un-compiled GCN model.
    """
    # Make default args
    model_default = {'input_embedd': {"input_node_vocab": 100, "input_edge_vocab": 10, "input_state_vocab": 100,
                                      "input_node_embedd": 64, "input_edge_embedd": 64, "input_state_embedd": 64,
                                      "input_tensor_type": 'ragged'},
                     'output_embedd': {"output_mode": 'graph', "output_tensor_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'linear']},
                     'gin_args': {"units": [64, 64], "use_bias": True, "activation": ['relu', 'linear'],
                                  "pooling_method": 'sum',
                                  "is_sorted": False, "has_unconnected": True}
                     }

    # Update model parameter
    input_embedd = update_model_args(model_default['input_embedd'], input_embedd)
    output_embedd = update_model_args(model_default['output_embedd'], output_embedd)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    gin_args = update_model_args(model_default['gin_args'], gin_args)

    # Make input embedding, if no feature dimension
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    edge_index_input = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)
    n = generate_node_embedding(node_input, input_node_shape, **input_embedd)
    edi = edge_index_input

    # Map to the required number of units. Not used in original paper.
    # n = Dense(gin_args["units"][0], use_bias=True, activation='linear')(n)
    # n = MLP(gin_args["units"], use_bias=gin_args["use_bias"], activation=gin_args["activation"])(n)

    list_embeddings = [n]
    for i in range(0, depth):
        n = GIN(**gin_args)([n, edi])
        n = BatchNormalization()(n)
        n = Activation("relu")(n)
        list_embeddings.append(n)

    if output_embedd["output_mode"] == "graph":
        out = [PoolingNodes()(x) for x in list_embeddings]  # will return tensor
        out = [MLP(**output_mlp)(x) for x in out]
        out = [Dropout(dropout)(x) for x in out]
        out = ks.layers.Add()(out)
        out = ks.layers.Activation(output_activation)(out)

    else:  # Node labeling
        out = n
        out = MLP(**output_mlp)(out)
        out = Activation(output_activation)(out)
        out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(
            out)  # no ragged for distribution supported atm

    model = ks.models.Model(inputs=[node_input, edge_index_input], outputs=out)

    return model
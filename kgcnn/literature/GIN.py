import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv.gin_conv import GIN
from kgcnn.layers.keras import Dropout, Activation, Dense
from kgcnn.layers.mlp import MLP, BatchNormMLP
from kgcnn.layers.pool.pooling import PoolingNodes
from kgcnn.utils.models import update_model_kwargs, generate_embedding

# How Powerful are Graph Neural Networks?
# Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka
# https://arxiv.org/abs/1810.00826

model_default = {'name': "GIN",
                 'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                 'input_embedding': {"node": {"input_dim": 95, "output_dim": 64}},
                 'output_embedding': 'graph',
                 'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                "activation": ['relu', 'relu', 'linear']},
                 'gin_args': {"units": [64, 64], "use_bias": True, "activation": ['relu', 'linear']},
                 'depth': 3, "output_activation": "softmax", "dropout": 0.0, 'verbose': 1,
                 }


@update_model_kwargs(model_default)
def make_model(inputs=None,
               input_embedding=None,
               output_embedding=None,
               output_mlp=None,
               depth=None,
               gin_args=None,
               output_activation=None,
               dropout=None, **kwargs):
    """Make GIN graph network via functional API. Default parameters can be found in :obj:`model_default`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in `Embedding` layers.
        output_embedding (str): Main embedding task for graph network. Either "node", ("edge") or "graph".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification `MLP` layer block.
            Defines number of model outputs and activation.
        depth (int): Number of graph embedding units or depth of the network.
        gin_args (dict): Dictionary of layer arguments unpacked in `GIN` convolutional layer.
        output_activation (str, dict): Activation for final output layer.
        dropout (float): Dropout to use.

    Returns:
        tf.keras.models.Model
    """

    # Make input
    assert len(inputs) == 2
    node_input = ks.layers.Input(**inputs[0])
    edge_index_input = ks.layers.Input(**inputs[1])

    # Embedding, if no feature dimension
    n = generate_embedding(node_input, inputs[0]['shape'], input_embedding['node'])
    edi = edge_index_input

    # Model
    # Map to the required number of units.
    n = Dense(gin_args["units"][0], use_bias=True, activation='linear')(n)
    # n = MLP(gin_args["units"], use_bias=gin_args["use_bias"], activation=gin_args["activation"])(n)
    list_embeddings = [n]
    for i in range(0, depth):
        n = GIN()([n, edi])
        n = BatchNormMLP(**gin_args)(n)
        list_embeddings.append(n)

    # Output embedding choice
    if output_embedding == "graph":
        out = [PoolingNodes()(x) for x in list_embeddings]  # will return tensor
        out = [MLP(**output_mlp)(x) for x in out]
        out = [Dropout(dropout)(x) for x in out]
        out = ks.layers.Add()(out)
        out = ks.layers.Activation(output_activation)(out)
    elif output_embedding == "node":  # Node labeling
        out = n
        out = MLP(**output_mlp)(out)
        out = Activation(output_activation)(out)
        out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(
            out)  # no ragged for distribution supported atm
    else:
        raise ValueError("Unsupported graph embedding for mode `GIN`")

    model = ks.models.Model(inputs=[node_input, edge_index_input], outputs=out)
    return model

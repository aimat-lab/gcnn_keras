import tensorflow.keras as ks

from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv.gin_conv import GIN
from kgcnn.layers.modules import DenseEmbedding, OptionalInputEmbedding
from kgcnn.layers.mlp import GraphMLP, MLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.utils.models import update_model_kwargs

# How Powerful are Graph Neural Networks?
# Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka
# https://arxiv.org/abs/1810.00826

model_default = {'name': "GIN",
                 'inputs': [{'shape': (None,), 'name': "node_attributes", 'dtype': 'float32', 'ragged': True},
                            {'shape': (None, 2), 'name': "edge_indices", 'dtype': 'int64', 'ragged': True}],
                 'input_embedding': {"node": {"input_dim": 95, "output_dim": 64}},
                 'gin_args': {"units": [64, 64], "use_bias": True, "activation": ['relu', 'linear'],
                              "use_normalization": True, "normalization_technique": "batch"},
                 'depth': 3, "dropout": 0.0, 'verbose': 10,
                 'last_mlp': {"use_bias": [True, True, True], "units": [64, 64, 64],
                              "activation": ['relu', 'relu', 'linear']},
                 'output_embedding': 'graph',
                 'output_mlp': {"use_bias": True, "units": 1,
                                "activation": "softmax"}
                 }


@update_model_kwargs(model_default)
def make_model(inputs=None,
               input_embedding=None,
               depth=None,
               gin_args=None,
               last_mlp=None,
               dropout=None,
               name=None,
               verbose=None,
               output_embedding=None,
               output_mlp=None
               ):
    """Make GIN graph network via functional API. Default parameters can be found in :obj:`model_default`.

    Args:
        inputs (list): List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding (dict): Dictionary of embedding arguments for nodes etc. unpacked in `Embedding` layers.
        depth (int): Number of graph embedding units or depth of the network.
        gin_args (dict): Dictionary of layer arguments unpacked in `GIN` convolutional layer.
        last_mlp (dict): Dictionary of layer arguments unpacked in last `MLP` layer before output or pooling.
        dropout (float): Dropout to use.
        name (str): Name of the model.
        verbose (int): Level of print output.
        output_embedding (str): Main embedding task for graph network. Either "node", ("edge") or "graph".
        output_mlp (dict): Dictionary of layer arguments unpacked in the final classification `MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        tf.keras.models.Model
    """

    # Make input
    assert len(inputs) == 2
    node_input = ks.layers.Input(**inputs[0])
    edge_index_input = ks.layers.Input(**inputs[1])

    # Embedding, if no feature dimension
    n = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)
    edi = edge_index_input

    # Model
    # Map to the required number of units.
    n = DenseEmbedding(gin_args["units"][0], use_bias=True, activation='linear')(n)
    # n = MLP(gin_args["units"], use_bias=gin_args["use_bias"], activation=gin_args["activation"])(n)
    list_embeddings = [n]
    for i in range(0, depth):
        n = GIN()([n, edi])
        n = GraphMLP(**gin_args)(n)
        list_embeddings.append(n)

    # Output embedding choice
    if output_embedding == "graph":
        out = [PoolingNodes()(x) for x in list_embeddings]  # will return tensor
        out = [MLP(**last_mlp)(x) for x in out]
        out = [ks.layers.Dropout(dropout)(x) for x in out]
        out = ks.layers.Add()(out)
        out = MLP(**output_mlp)(out)
    elif output_embedding == "node":  # Node labeling
        out = n
        out = GraphMLP(**last_mlp)(out)
        out = GraphMLP(**output_mlp)(out)
        # no ragged for distribution supported atm
        out = ChangeTensorType(input_tensor_type='ragged', output_tensor_type="tensor")(out)
    else:
        raise ValueError("Unsupported graph embedding for mode `GIN`")

    model = ks.models.Model(inputs=[node_input, edge_index_input], outputs=out)
    return model

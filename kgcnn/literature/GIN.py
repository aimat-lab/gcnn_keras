import tensorflow.keras as ks
import pprint

from kgcnn.utils.models import update_model_args
from kgcnn.utils.models import generate_node_embedding
from kgcnn.layers.keras import Dropout, Activation
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.mlp import MLP, BatchNormMLP
from kgcnn.layers.casting import ChangeTensorType
from kgcnn.layers.conv import GIN


# How Powerful are Graph Neural Networks?
# Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka
# https://arxiv.org/abs/1810.00826


def make_gin(**kwargs):
    """Make GCN model.

    Args:
        **kwargs

    Returns:
        tf.keras.models.Model: Un-compiled GCN model.
    """
    model_args = kwargs
    model_default = {'input_node_shape': None,
                     'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 64},
                                         "edges": {"input_dim": 10, "output_dim": 64},
                                         "state": {"input_dim": 100, "output_dim": 64}},
                     'output_embedding': {"output_mode": 'graph', "output_tensor_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True, False], "units": [25, 10, 1],
                                    "activation": ['relu', 'relu', 'linear']},
                     'gin_args': {"units": [64, 64], "use_bias": True, "activation": ['relu', 'linear']},
                     'depth': 3, "output_activation": "softmax", "dropout": 0.0, 'verbose': 1
                     }
    m = update_model_args(model_default, model_args)
    if m['verbose'] > 0:
        print("INFO: Updated functional make model kwargs:")
        pprint.pprint(m)

    # Update model parameter
    input_embedding = m['input_embedding']
    output_embedding = m['output_embedding']
    output_mlp = m['output_mlp']
    depth = m['depth']
    input_node_shape = m['input_node_shape']
    gin_args = m['gin_args']
    output_activation = m['output_activation']
    dropout = m['dropout']

    # Make input
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    edge_index_input = ks.layers.Input(shape=(None, 2), name='edge_index_input', dtype="int64", ragged=True)

    # Embedding, if no feature dimension
    n = generate_node_embedding(node_input, input_node_shape, input_embedding['nodes'])
    edi = edge_index_input

    # Model
    # Map to the required number of units.
    # n = Dense(gin_args["units"][0], use_bias=True, activation='linear')(n)
    # n = MLP(gin_args["units"], use_bias=gin_args["use_bias"], activation=gin_args["activation"])(n)
    list_embeddings = [n]
    for i in range(0, depth):
        n = GIN()([n, edi])
        n = BatchNormMLP(**gin_args)(n)
        list_embeddings.append(n)

    # Output embedding choice
    if output_embedding["output_mode"] == "graph":
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